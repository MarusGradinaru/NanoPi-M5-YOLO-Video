#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "secondary_classifier.h"
#include "imagenet_to_coco80.h"
#include "file_utils.h"
#include "image_utils.h"

static void softmax(float *a, int n) {
  if (n <= 0) return;
  float maxv = a[0];
  for (int i = 1; i < n; i++) if (a[i] > maxv) maxv = a[i];
  float sum = 0.0f;
  for (int i = 0; i < n; i++) { a[i] = expf(a[i] - maxv); sum += a[i]; }
  if (sum <= 0.0f) return;
  for (int i = 0; i < n; i++) a[i] /= sum;
}

int init_secondary_classifier(const char *model_path, secondary_classifier_context_t *ctx) {
  if (ctx == nullptr || model_path == nullptr) return -1;
  memset(ctx, 0, sizeof(*ctx));

  int model_len = 0;
  char *model = nullptr;
  model_len = read_data_from_file(model_path, &model);
  if (model == nullptr || model_len <= 0) {
    printf("Secondary classifier: failed to load model: %s\n", model_path);
    return -1;
  }

  int ret = rknn_init(&ctx->rknn_ctx, model, model_len, 0, nullptr);
  free(model);
  if (ret < 0) {
    printf("Secondary classifier: rknn_init failed! ret=%d\n", ret);
    ctx->rknn_ctx = 0;
    return ret;
  }

  ret = rknn_query(ctx->rknn_ctx, RKNN_QUERY_IN_OUT_NUM, &ctx->io_num, sizeof(ctx->io_num));
  if (ret != RKNN_SUCC || ctx->io_num.n_input != 1 || ctx->io_num.n_output != 1) {
    printf("Secondary classifier: unexpected model I/O (%d input, %d output), ret=%d\n",
      ctx->io_num.n_input, ctx->io_num.n_output, ret);
    release_secondary_classifier(ctx);
    return -1;
  }

  ctx->input_attrs = (rknn_tensor_attr*)calloc(ctx->io_num.n_input, sizeof(rknn_tensor_attr));
  ctx->output_attrs = (rknn_tensor_attr*)calloc(ctx->io_num.n_output, sizeof(rknn_tensor_attr));
  if (ctx->input_attrs == nullptr || ctx->output_attrs == nullptr) {
    printf("Secondary classifier: failed to allocate tensor attributes\n");
    release_secondary_classifier(ctx);
    return -1;
  }

  for (int i = 0; i < ctx->io_num.n_input; i++) {
    ctx->input_attrs[i].index = i;
    ret = rknn_query(ctx->rknn_ctx, RKNN_QUERY_INPUT_ATTR, &ctx->input_attrs[i], sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      printf("Secondary classifier: input attr query failed! ret=%d\n", ret);
      release_secondary_classifier(ctx);
      return ret;
    }
  }

  for (int i = 0; i < ctx->io_num.n_output; i++) {
    ctx->output_attrs[i].index = i;
    ret = rknn_query(ctx->rknn_ctx, RKNN_QUERY_OUTPUT_ATTR, &ctx->output_attrs[i], sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      printf("Secondary classifier: output attr query failed! ret=%d\n", ret);
      release_secondary_classifier(ctx);
      return ret;
    }
  }

  const rknn_tensor_attr &in = ctx->input_attrs[0];
  if (in.fmt == RKNN_TENSOR_NCHW) {
    ctx->model_channel = in.dims[1];
    ctx->model_height = in.dims[2];
    ctx->model_width = in.dims[3];
  } else {
    ctx->model_height = in.dims[1];
    ctx->model_width = in.dims[2];
    ctx->model_channel = in.dims[3];
  }

  if (ctx->output_attrs[0].n_elems != 1000) {
    printf("Secondary classifier: expected 1000 ImageNet outputs, got %d\n", ctx->output_attrs[0].n_elems);
    release_secondary_classifier(ctx);
    return -1;
  }

  printf("Secondary classifier : %s\n", model_path);
  printf("Secondary input      : %dx%dx%d\n", ctx->model_width, ctx->model_height, ctx->model_channel);
  return 0;
}

int classify_secondary(secondary_classifier_context_t *ctx, image_buffer_t *src_img, secondary_classifier_result_t *result) {
  if (ctx == nullptr || src_img == nullptr || result == nullptr || ctx->rknn_ctx == 0) return -1;
  result->coco_class = -1;
  result->score = 0.0f;
  result->mapped_mass = 0.0f;

  int ret = 0;
  image_buffer_t img;
  memset(&img, 0, sizeof(img));
  img.width = ctx->model_width;
  img.height = ctx->model_height;
  img.format = IMAGE_FORMAT_RGB888;
  img.size = get_image_size(&img);
  img.virt_addr = (unsigned char*)malloc(img.size);
  if (img.virt_addr == nullptr) return -1;

  ret = convert_image(src_img, &img, nullptr, nullptr, 0);
  if (ret < 0) {
    printf("Secondary classifier: convert_image failed! ret=%d\n", ret);
    free(img.virt_addr);
    return ret;
  }

  rknn_input input;
  memset(&input, 0, sizeof(input));
  input.index = 0;
  input.type = RKNN_TENSOR_UINT8;
  input.fmt = RKNN_TENSOR_NHWC;
  input.size = ctx->model_width * ctx->model_height * ctx->model_channel;
  input.buf = img.virt_addr;

  ret = rknn_inputs_set(ctx->rknn_ctx, 1, &input);
  if (ret < 0) {
    printf("Secondary classifier: rknn_inputs_set failed! ret=%d\n", ret);
    free(img.virt_addr);
    return ret;
  }

  ret = rknn_run(ctx->rknn_ctx, nullptr);
  if (ret < 0) {
    printf("Secondary classifier: rknn_run failed! ret=%d\n", ret);
    free(img.virt_addr);
    return ret;
  }

  rknn_output output;
  memset(&output, 0, sizeof(output));
  output.want_float = 1;
  ret = rknn_outputs_get(ctx->rknn_ctx, 1, &output, nullptr);
  if (ret < 0) {
    printf("Secondary classifier: rknn_outputs_get failed! ret=%d\n", ret);
    free(img.virt_addr);
    return ret;
  }

  float *scores = (float*)output.buf;
  const int n = ctx->output_attrs[0].n_elems;
  softmax(scores, n);

  float coco_scores[80] = {0.0f};
  for (int i = 0; i < n; i++) {
    int coco_id = imagenet_to_coco80(i);
    if (coco_id < 0 || coco_id >= 80) continue;
    coco_scores[coco_id] += scores[i];
    result->mapped_mass += scores[i];
  }

  for (int cls = 0; cls < 80; cls++) {
    if (result->coco_class < 0 || coco_scores[cls] > result->score) {
      result->coco_class = cls;
      result->score = coco_scores[cls];
    }
  }

  rknn_outputs_release(ctx->rknn_ctx, 1, &output);
  free(img.virt_addr);
  return 0;
}

int release_secondary_classifier(secondary_classifier_context_t *ctx) {
  if (ctx == nullptr) return 0;
  if (ctx->input_attrs != nullptr) { free(ctx->input_attrs); ctx->input_attrs = nullptr; }
  if (ctx->output_attrs != nullptr) { free(ctx->output_attrs); ctx->output_attrs = nullptr; }
  if (ctx->rknn_ctx != 0) { rknn_destroy(ctx->rknn_ctx); ctx->rknn_ctx = 0; }
  return 0;
}
