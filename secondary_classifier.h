#pragma once

#include "rknn_api.h"
#include "common.h"

typedef struct {
  rknn_context rknn_ctx;
  rknn_input_output_num io_num;
  rknn_tensor_attr *input_attrs;
  rknn_tensor_attr *output_attrs;
  int model_channel;
  int model_width;
  int model_height;
} secondary_classifier_context_t;

typedef struct {
  int coco_class;
  float score;
  float mapped_mass;
} secondary_classifier_result_t;

int init_secondary_classifier(const char *model_path, secondary_classifier_context_t *ctx);
int classify_secondary(secondary_classifier_context_t *ctx, image_buffer_t *src_img, secondary_classifier_result_t *result);
int release_secondary_classifier(secondary_classifier_context_t *ctx);
