#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <strings.h>
#include <string>
#include <deque>
#include <vector>
#include <limits>
#include <map>
#include <algorithm>
#include <utility>
#include <cmath>
#include <filesystem>

#include <opencv2/opencv.hpp>

#include "yolov8.h"
#include "image_utils.h"
#include "secondary_classifier.h"
#include "imagenet_to_coco80.h"

struct RawDetection {
  int cls_id;
  float confidence;
  cv::Rect2f box;
};

struct TrackSample {
  int cls_id;
  float confidence;
  bool detected;
};

struct TrackEval {
  int best_class = -1;
  int class_count = 0;
  int longest_consecutive = 0;
  float avg_confidence = 0.0f;
  bool full_window = false;
  bool presence_ok = false;
  bool consecutive_ok = false;
  bool confidence_ok = false;
  bool confirmed = false;
};

struct Track {
  int id = 0;
  cv::Rect2f box;
  cv::KalmanFilter kf;
  std::deque<TrackSample> history;
  std::map<int, unsigned long long> raw_class_counts;
  std::map<int, double> raw_conf_sums;
  int confirmed_class = -1;
  float confirmed_avg_confidence = 0.0f;
  int missed_frames = 0;
  int age_frames = 0;
  bool matched_this_frame = false;
  bool ever_full_window = false;
  bool ever_presence_ok = false;
  bool ever_consecutive_ok = false;
  bool ever_confidence_ok = false;
  bool secondary_pending = false;
  bool secondary_checked = false;
  unsigned long long secondary_next_retry_frame = 0;
  int secondary_retry_count = 0;
  int secondary_class = -1;
  float secondary_score = 0.0f;
  float secondary_mapped_mass = 0.0f;
  int final_class = -1;
};


struct TrackColor {
  cv::Scalar bg;
  cv::Scalar text;
};

static const TrackColor TRACK_COLORS[] = {
  {cv::Scalar(255,   0,   0), cv::Scalar(255, 255, 255)},  // blue
  {cv::Scalar(  0,   0, 255), cv::Scalar(255, 255, 255)},  // red
  {cv::Scalar(  0, 200,   0), cv::Scalar(  0,   0,   0)},  // green
  {cv::Scalar(  0, 220, 255), cv::Scalar(  0,   0,   0)},  // yellow
  {cv::Scalar(180,   0, 180), cv::Scalar(255, 255, 255)},  // violet
  {cv::Scalar(  0, 140, 255), cv::Scalar(  0,   0,   0)},  // orange
  {cv::Scalar(255, 255,   0), cv::Scalar(  0,   0,   0)},  // cyan
  {cv::Scalar(255,   0, 255), cv::Scalar(255, 255, 255)}   // magenta
};

static inline const TrackColor& track_color(int track_id) {
  constexpr int count = static_cast<int>(sizeof(TRACK_COLORS) / sizeof(TRACK_COLORS[0]));
  int index = track_id > 0 ? (track_id - 1) % count : 0;
  return TRACK_COLORS[index];
}


static float box_iou(const cv::Rect2f &a, const cv::Rect2f &b) {
  cv::Rect2f inter = a & b;
  float inter_area = inter.area();
  float union_area = a.area() + b.area() - inter_area;
  return union_area > 0.0f ? inter_area / union_area : 0.0f;
}

static cv::Rect2f state_to_box(const cv::Mat &state) {
  float cx = state.at<float>(0), cy = state.at<float>(1);
  float w = std::max(2.0f, state.at<float>(2)), h = std::max(2.0f, state.at<float>(3));
  return cv::Rect2f(cx - w * 0.5f, cy - h * 0.5f, w, h);
}

static void init_kalman(Track &track, const cv::Rect2f &box) {
  track.kf.init(8, 4, 0, CV_32F);
  track.kf.transitionMatrix = (cv::Mat_<float>(8, 8) <<
    1,0,0,0,1,0,0,0,
    0,1,0,0,0,1,0,0,
    0,0,1,0,0,0,1,0,
    0,0,0,1,0,0,0,1,
    0,0,0,0,1,0,0,0,
    0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,1);
  track.kf.measurementMatrix = cv::Mat::zeros(4, 8, CV_32F);
  for (int i = 0; i < 4; i++) track.kf.measurementMatrix.at<float>(i, i) = 1.0f;
  cv::setIdentity(track.kf.processNoiseCov, cv::Scalar::all(1e-2));
  cv::setIdentity(track.kf.measurementNoiseCov, cv::Scalar::all(1e-1));
  cv::setIdentity(track.kf.errorCovPost, cv::Scalar::all(1.0));
  track.kf.statePost = cv::Mat::zeros(8, 1, CV_32F);
  track.kf.statePost.at<float>(0) = box.x + box.width * 0.5f;
  track.kf.statePost.at<float>(1) = box.y + box.height * 0.5f;
  track.kf.statePost.at<float>(2) = box.width;
  track.kf.statePost.at<float>(3) = box.height;
  track.box = box;
}

static void predict_track(Track &track) {
  track.box = state_to_box(track.kf.predict());
}

static void update_track(Track &track, const RawDetection &det) {
  cv::Mat measurement(4, 1, CV_32F);
  measurement.at<float>(0) = det.box.x + det.box.width * 0.5f;
  measurement.at<float>(1) = det.box.y + det.box.height * 0.5f;
  measurement.at<float>(2) = det.box.width;
  measurement.at<float>(3) = det.box.height;
  track.box = state_to_box(track.kf.correct(measurement));
  track.history.back() = {det.cls_id, det.confidence, true};
  track.raw_class_counts[det.cls_id]++;
  track.raw_conf_sums[det.cls_id] += det.confidence;
  track.matched_this_frame = true;
  track.missed_frames = 0;
}

static std::vector<int> hungarian(const std::vector<std::vector<float>> &cost) {
  int rows = static_cast<int>(cost.size());
  int cols = rows ? static_cast<int>(cost[0].size()) : 0;
  if (!rows || !cols) return std::vector<int>(rows, -1);
  int n = std::max(rows, cols);
  std::vector<std::vector<float>> a(n + 1, std::vector<float>(n + 1, 1.0f));
  for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) a[i + 1][j + 1] = cost[i][j];

  std::vector<float> u(n + 1), v(n + 1);
  std::vector<int> p(n + 1), way(n + 1);
  for (int i = 1; i <= n; i++) {
    p[0] = i;
    int j0 = 0;
    std::vector<float> minv(n + 1, std::numeric_limits<float>::max());
    std::vector<char> used(n + 1, false);
    do {
      used[j0] = true;
      int i0 = p[j0], j1 = 0;
      float delta = std::numeric_limits<float>::max();
      for (int j = 1; j <= n; j++) if (!used[j]) {
        float cur = a[i0][j] - u[i0] - v[j];
        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
      }
      for (int j = 0; j <= n; j++) {
        if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
          else minv[j] -= delta;
      }
      j0 = j1;
    } while (p[j0] != 0);
    do {
      int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 != 0);
  }

  std::vector<int> assignment(rows, -1);
  for (int j = 1; j <= n; j++) if (p[j] >= 1 && p[j] <= rows && j <= cols) assignment[p[j] - 1] = j - 1;
  return assignment;
}

struct MatchResult {
  std::vector<std::pair<int, int>> matches;
  std::vector<int> unmatched_tracks;
  std::vector<int> unmatched_detections;
};

static MatchResult match_by_iou(const std::vector<Track> &tracks, const std::vector<RawDetection> &detections,
  const std::vector<int> &track_indices, const std::vector<int> &det_indices, float min_iou) {
  MatchResult result;
  if (track_indices.empty()) { result.unmatched_detections = det_indices; return result; }
  if (det_indices.empty()) { result.unmatched_tracks = track_indices; return result; }

  std::vector<std::vector<float>> cost(track_indices.size(), std::vector<float>(det_indices.size(), 1.0f));
  for (int i = 0; i < static_cast<int>(track_indices.size()); i++)
    for (int j = 0; j < static_cast<int>(det_indices.size()); j++)
      cost[i][j] = 1.0f - box_iou(tracks[track_indices[i]].box, detections[det_indices[j]].box);

  std::vector<int> assignment = hungarian(cost);
  std::vector<bool> det_used(det_indices.size(), false);
  for (int i = 0; i < static_cast<int>(track_indices.size()); i++) {
    int j = assignment[i];
    if (j >= 0 && j < static_cast<int>(det_indices.size()) && !det_used[j]) {
      float iou = 1.0f - cost[i][j];
      if (iou >= min_iou) { result.matches.push_back({track_indices[i], det_indices[j]}); det_used[j] = true; continue; }
    }
    result.unmatched_tracks.push_back(track_indices[i]);
  }
  for (int j = 0; j < static_cast<int>(det_indices.size()); j++)
    if (!det_used[j]) result.unmatched_detections.push_back(det_indices[j]);
  return result;
}

static TrackEval evaluate_track(const Track &track, int window_frames, float track_ratio, float consecutive_ratio,
  float avg_conf_threshold) {
  TrackEval e;
  if (static_cast<int>(track.history.size()) < window_frames) return e;
  e.full_window = true;

  std::map<int, int> counts;
  std::map<int, double> conf_sums;
  for (const auto &s : track.history) if (s.detected) { counts[s.cls_id]++; conf_sums[s.cls_id] += s.confidence; }
  for (const auto &entry : counts) {
    if (e.best_class < 0 || entry.second > e.class_count ||
      (entry.second == e.class_count && conf_sums[entry.first] > conf_sums[e.best_class])) {
      e.best_class = entry.first; e.class_count = entry.second;
    }
  }
  if (e.best_class < 0) return e;

  int run = 0;
  for (const auto &s : track.history) {
    if (s.detected && s.cls_id == e.best_class) { run++; e.longest_consecutive = std::max(e.longest_consecutive, run); }
      else run = 0;
  }

  e.avg_confidence = static_cast<float>(conf_sums[e.best_class] / e.class_count);
  int required_presence = static_cast<int>(std::ceil(window_frames * track_ratio));
  int required_consecutive = static_cast<int>(std::ceil(window_frames * consecutive_ratio));
  e.presence_ok = e.class_count >= required_presence;
  e.consecutive_ok = e.longest_consecutive >= required_consecutive;
  e.confidence_ok = e.avg_confidence >= avg_conf_threshold;
  e.confirmed = e.presence_ok && e.consecutive_ok && e.confidence_ok;
  return e;
}

static int best_lifetime_class(const Track &track) {
  int best_class = -1; unsigned long long best_count = 0; double best_conf = 0.0;
  for (const auto &entry : track.raw_class_counts) {
    double conf = track.raw_conf_sums.count(entry.first) ? track.raw_conf_sums.at(entry.first) : 0.0;
    if (best_class < 0 || entry.second > best_count || (entry.second == best_count && conf > best_conf)) {
      best_class = entry.first; best_count = entry.second; best_conf = conf;
    }
  }
  return best_class;
}

static bool parse_int_parameter(const char *arg, const char *name, long min_value, long max_value, long &value) {
  size_t len = strlen(name);
  if (strncmp(arg, name, len) != 0 || arg[len] != ':') return false;
  const char *value_str = arg + len + 1;
  if (*value_str == '\0') return false;
  char *endptr = nullptr;
  long parsed = strtol(value_str, &endptr, 10);
  if (*endptr != '\0' || parsed < min_value || parsed > max_value) return false;
  value = parsed;
  return true;
}

static void print_usage(const char *prog) {
  printf("Usage: %s <video_file> [yolo:<model.rknn>] [rnet:<model.rknn>] [no-zc] [ct:1-100] "
    "[ht:1-100] [tw:200-5000] [tr:1-100] [tc:1-100] [ac:1-100] [rs:1-100]\n", prog);
  printf("  yolo  = main YOLO RKNN model filename (default = yolov8s_rk3576_i8.rknn)\n");
  printf("  rnet  = secondary ResNet50 RKNN model filename (default = resnet50-v2-7-i8.rknn)\n");
  printf("  no-zc = disable zero-copy backend (zero-copy is enabled by default)\n");
  printf("  ct    = minimum raw confidence accepted by ByteTrack %% (default = 40)\n");
  printf("  ht    = ByteTrack high-confidence threshold %% (default = 55)\n");
  printf("  tw    = tracking window in milliseconds (default = 1000)\n");
  printf("  tr    = required same-class presence in the whole window %% (default = 80)\n");
  printf("  tc    = required consecutive detections of the same class %% (default = 30)\n");
  printf("  ac    = required average confidence of the confirmed class %% (default = 65)\n");
  printf("  rs    = minimum ResNet score required to override YOLO %% (default = 80)\n");
  printf("Example: %s test.mp4 ct:40 ht:55 tw:1000 tr:80 tc:30 ac:65 rs:80\n", prog);
}

int main(int argc, char **argv) {
    if (argc < 2) { print_usage(argv[0]); return -1; }

    bool use_zero_copy = true;
    float conf_thresh = 0.40f;
    float high_thresh = 0.55f;
    int track_ms = 1000;
    float track_ratio = 0.80f;
    float consecutive_ratio = 0.30f;
    float avg_conf_threshold = 0.65f;
    float resnet_override_threshold = 0.80f;
    const float BT_MATCH_IOU_HIGH = 0.30f;
    const float BT_MATCH_IOU_LOW = 0.20f;
    const unsigned long long SECONDARY_RETRY_FRAMES = 10;
    const int ADVISORY_MAX_RETRIES = 2;
    std::string video_src_path = argv[1];
    std::string yolo_model_file = "yolov8s_rk3576_i8.rknn";
    std::string resnet_model_file = "resnet50-v2-7-i8.rknn";

    for (int i = 2; i < argc; i++) {
      if (strcmp(argv[i], "no-zc") == 0)
        { use_zero_copy = false; continue; }
      if (strncmp(argv[i], "yolo:", 5) == 0 && argv[i][5] != '\0')
        { yolo_model_file = argv[i] + 5; continue; }
      if (strncmp(argv[i], "rnet:", 5) == 0 && argv[i][5] != '\0')
        { resnet_model_file = argv[i] + 5; continue; }
      long value = 0;
      if (parse_int_parameter(argv[i], "ct", 1, 100, value))
        { conf_thresh = static_cast<float>(value) / 100.0f; continue; }
      if (parse_int_parameter(argv[i], "ht", 1, 100, value))
        { high_thresh = static_cast<float>(value) / 100.0f; continue; }
      if (parse_int_parameter(argv[i], "tw", 200, 5000, value))
        { track_ms = static_cast<int>(value); continue; }
      if (parse_int_parameter(argv[i], "tr", 1, 100, value))
        { track_ratio = static_cast<float>(value) / 100.0f; continue; }
      if (parse_int_parameter(argv[i], "tc", 1, 100, value))
        { consecutive_ratio = static_cast<float>(value) / 100.0f; continue; }
      if (parse_int_parameter(argv[i], "ac", 1, 100, value))
        { avg_conf_threshold = static_cast<float>(value) / 100.0f; continue; }
      if (parse_int_parameter(argv[i], "rs", 1, 100, value))
        { resnet_override_threshold = static_cast<float>(value) / 100.0f; continue; }
      printf("Invalid or unknown parameter: %s\n\n", argv[i]); print_usage(argv[0]); return -1;
    }

    if (high_thresh < conf_thresh) { printf("Invalid thresholds: ht must be >= ct.\n"); return -1; }
    auto valid_model_filename = [](const std::string &name) {
      std::filesystem::path p(name);
      return !name.empty() && p.filename() == p;
    };
    if (!valid_model_filename(yolo_model_file) || !valid_model_filename(resnet_model_file))
      { printf("yolo: and rnet: must contain a filename only, not a path.\n"); return -1; }

    std::error_code ec;
    std::filesystem::path exe_path = std::filesystem::canonical("/proc/self/exe", ec);
    if (ec) { printf("Failed to determine executable path: %s\n", ec.message().c_str()); return -1; }
    std::filesystem::path model_dir = exe_path.parent_path() / "model";
    std::string yolo_model_path = (model_dir / yolo_model_file).string();
    std::string resnet_model_path = (model_dir / resnet_model_file).string();

    int ret = 0;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_ctx));
    init_post_process();

    if (use_zero_copy) ret = init_yolov8_model_zc(yolo_model_path.c_str(), &rknn_app_ctx);
      else ret = init_yolov8_model(yolo_model_path.c_str(), &rknn_app_ctx);
    if (ret != 0) {
      printf("init_yolov8_model failed! ret=%d model=%s\n", ret, yolo_model_path.c_str());
      deinit_post_process();
      return -1;
    }

    secondary_classifier_context_t secondary_ctx;
    memset(&secondary_ctx, 0, sizeof(secondary_ctx));
    ret = init_secondary_classifier(resnet_model_path.c_str(), &secondary_ctx);
    if (ret != 0) {
      printf("init_secondary_classifier failed! ret=%d model=%s\n", ret, resnet_model_path.c_str());
      if (use_zero_copy) release_yolov8_model_zc(&rknn_app_ctx); else release_yolov8_model(&rknn_app_ctx);
      deinit_post_process();
      return -1;
    }

    cv::VideoCapture cap(video_src_path);
    if (!cap.isOpened()) {
      printf("Failed to open video: %s\n", video_src_path.c_str());
      release_secondary_classifier(&secondary_ctx);
      if (use_zero_copy) release_yolov8_model_zc(&rknn_app_ctx); else release_yolov8_model(&rknn_app_ctx);
      deinit_post_process();
      return -1;
    }

    const double video_fps = cap.get(cv::CAP_PROP_FPS);
    const int video_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int video_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    const double frame_count = cap.get(cv::CAP_PROP_FRAME_COUNT);
    if (!std::isfinite(video_fps) || video_fps <= 0.0) {
      printf("Invalid video FPS: %.3f. Tracking window can not be calculated.\n", video_fps);
      release_secondary_classifier(&secondary_ctx);
      if (use_zero_copy) release_yolov8_model_zc(&rknn_app_ctx); else release_yolov8_model(&rknn_app_ctx);
      deinit_post_process();
      return -1;
    }

    const int track_frames = std::max(1, static_cast<int>(std::lround(video_fps * track_ms / 1000.0)));
    const int required_presence = static_cast<int>(std::ceil(track_frames * track_ratio));
    const int required_consecutive = static_cast<int>(std::ceil(track_frames * consecutive_ratio));
    const int max_lost_frames = track_frames;

    printf("Inference backend    : %s\n", use_zero_copy ? "ZERO-COPY" : "NORMAL");
    printf("Confidence threshold : %.0f %%\n", conf_thresh * 100.0f);
    printf("ByteTrack high thr.  : %.0f %%\n", high_thresh * 100.0f);
    printf("Tracking window      : %d ms (%d frames)\n", track_ms, track_frames);
    printf("Track validation     : TR %.0f %% (%d/%d), TC %.0f %% (%d consecutive), AC %.0f %%\n",
      track_ratio * 100.0f, required_presence,
      track_frames, consecutive_ratio * 100.0f, required_consecutive, avg_conf_threshold * 100.0f);
    printf("ByteTrack IoU        : high %.2f, low %.2f\n", BT_MATCH_IOU_HIGH, BT_MATCH_IOU_LOW);
    printf("ResNet override thr. : %.0f %%\n", resnet_override_threshold * 100.0f);
    printf("ResNet retry interval: %llu frames\n", SECONDARY_RETRY_FRAMES);
    printf("Advisory max retries  : %d\n", ADVISORY_MAX_RETRIES);
    printf("\nVideo information:\n");
    printf("  resolution : %dx%d\n", video_width, video_height);
    printf("  FPS        : %.2f\n", video_fps);
    printf("  frames     : %.0f\n\n", frame_count);

    cv::namedWindow("YOLOv8 - RK3576 NPU", cv::WINDOW_NORMAL);
    cv::setWindowProperty("YOLOv8 - RK3576 NPU", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

    cv::Mat frame;
    constexpr int UI_REF_WIDTH = 1024;
    constexpr int UI_REF_HEIGHT = 600;
    const int screen_width = 1024;
    const int screen_height = 600;
    const double ui_scale = std::min(static_cast<double>(screen_width) / UI_REF_WIDTH,
      static_cast<double>(screen_height) / UI_REF_HEIGHT);
    auto ui_px = [&](int value) { return std::max(1, static_cast<int>(std::lround(value * ui_scale))); };
    unsigned long long frame_number = 0;

    struct PerfSample {
      std::chrono::steady_clock::time_point time;
      double npu_ms;
      double fps;
    };

    std::deque<PerfSample> perf_samples;
    constexpr double PERF_WINDOW_SECONDS = 5.0;

    std::vector<Track> tracks;
    int next_track_id = 1;
    bool show_statistics = false;
    bool show_info = false;
    int playback_delay_ms = 0;
    constexpr int PLAYBACK_DELAY_STEP_MS = 25;
    constexpr int PLAYBACK_DELAY_MAX_MS = 1000;
    constexpr double PLAYBACK_DELAY_OSD_SECONDS = 5.0;
    auto playback_delay_osd_until = std::chrono::steady_clock::time_point::min();
    unsigned long long candidate_tracks_created = 0;
    unsigned long long confirmed_tracks_total = 0;
    unsigned long long rejected_tracks_total = 0;
    unsigned long long reclassifications = 0;
    unsigned long long rejected_short_lived = 0;
    unsigned long long rejected_low_presence = 0;
    unsigned long long rejected_low_consecutive = 0;
    unsigned long long rejected_low_confidence = 0;
    std::map<int, unsigned long long> confirmed_objects;
    std::map<int, unsigned long long> rejected_classes;
    unsigned long long secondary_checks = 0;
    unsigned long long secondary_strict_checks = 0;
    unsigned long long secondary_advisory_checks = 0;
    unsigned long long secondary_agreements = 0;
    unsigned long long secondary_disagreements = 0;
    unsigned long long secondary_unmapped = 0;
    unsigned long long secondary_failures = 0;
    unsigned long long secondary_advisory_fallbacks = 0;
    std::map<std::pair<int, int>, unsigned long long> secondary_disagreement_pairs;
    unsigned long long secondary_overrides = 0;
    unsigned long long secondary_below_threshold = 0;
    std::map<std::pair<int, int>, unsigned long long> secondary_override_pairs;
    std::map<int, unsigned long long> final_objects;

    auto count_rejected_track = [&](const Track &track) {
      rejected_tracks_total++;
      int cls = best_lifetime_class(track);
      if (cls >= 0) rejected_classes[cls]++;
      if (!track.ever_full_window) { rejected_short_lived++; return; }
      if (!track.ever_presence_ok) rejected_low_presence++;
      if (!track.ever_consecutive_ok) rejected_low_consecutive++;
      if (!track.ever_confidence_ok) rejected_low_confidence++;
    };

    auto set_final_class = [&](Track &track, int cls_id) {
      if (track.final_class == cls_id) return;
      if (track.final_class >= 0) {
        auto it = final_objects.find(track.final_class);
        if (it != final_objects.end()) { if (it->second > 1) it->second--; else final_objects.erase(it); }
      }
      track.final_class = cls_id;
      if (cls_id >= 0) final_objects[cls_id]++;
    };

    auto set_confirmed_class = [&](Track &track, int cls_id) {
      if (track.confirmed_class == cls_id) return;
      if (track.confirmed_class < 0) confirmed_tracks_total++;
      else {
        auto it = confirmed_objects.find(track.confirmed_class);
        if (it != confirmed_objects.end()) { if (it->second > 1) it->second--; else confirmed_objects.erase(it); }
        reclassifications++;
      }
      track.confirmed_class = cls_id;
      confirmed_objects[cls_id]++;

      track.secondary_class = -1;
      track.secondary_score = 0.0f;
      track.secondary_mapped_mass = 0.0f;
      track.secondary_next_retry_frame = 0;
      track.secondary_retry_count = 0;

      ResNetValidationMode mode = coco_resnet_validation_mode(cls_id);
      if (mode == RESNET_NONE) {
        set_final_class(track, cls_id);
        track.secondary_pending = false;
        track.secondary_checked = false;
      } else {
        set_final_class(track, -1);
        track.secondary_pending = true;
        track.secondary_checked = false;
      }
    };

    while (cap.read(frame)) {  // ---------- Frame Loop ----------
      frame_number++;
      if (frame.empty()) break;
      auto frame_start = std::chrono::steady_clock::now();

      image_buffer_t src_image;
      memset(&src_image, 0, sizeof(src_image));
      src_image.width = frame.cols;
      src_image.height = frame.rows;
      src_image.width_stride = frame.cols;
      src_image.height_stride = frame.rows;
      src_image.format = IMAGE_FORMAT_RGB888;
      src_image.virt_addr = frame.data;
      src_image.size = frame.total() * frame.elemSize();

      cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

      object_detect_result_list od_results;
      memset(&od_results, 0, sizeof(od_results));

      auto inference_start = std::chrono::steady_clock::now();
      if (use_zero_copy) ret = inference_yolov8_model_zc(&rknn_app_ctx, &src_image, &od_results, conf_thresh);
        else ret = inference_yolov8_model(&rknn_app_ctx, &src_image, &od_results, conf_thresh);
      auto inference_end = std::chrono::steady_clock::now();
      if (ret != 0) { printf("inference_yolov8_model failed! ret=%d\n", ret); break; }

      cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);

      std::vector<RawDetection> detections;
      detections.reserve(od_results.count);
      for (int i = 0; i < od_results.count; i++) {
        object_detect_result *det = &od_results.results[i];
        int x1 = std::max(0, std::min(det->box.left, frame.cols - 1));
        int y1 = std::max(0, std::min(det->box.top, frame.rows - 1));
        int x2 = std::max(0, std::min(det->box.right, frame.cols - 1));
        int y2 = std::max(0, std::min(det->box.bottom, frame.rows - 1));
        if (x2 <= x1 || y2 <= y1) continue;
        detections.push_back({det->cls_id, det->prop,
          cv::Rect2f(static_cast<float>(x1), static_cast<float>(y1), static_cast<float>(x2 - x1 + 1),
            static_cast<float>(y2 - y1 + 1))});
      }

      for (auto &track : tracks) {
        track.history.push_back({-1, 0.0f, false});
        while (static_cast<int>(track.history.size()) > track_frames) track.history.pop_front();
        track.matched_this_frame = false;
        track.missed_frames++;
        track.age_frames++;
        predict_track(track);
      }

      std::vector<int> high_dets, low_dets, all_tracks;
      for (int i = 0; i < static_cast<int>(detections.size()); i++) {
        if (detections[i].confidence >= high_thresh) high_dets.push_back(i); else low_dets.push_back(i);
      }
      for (int i = 0; i < static_cast<int>(tracks.size()); i++) all_tracks.push_back(i);

      MatchResult high_match = match_by_iou(tracks, detections, all_tracks, high_dets, BT_MATCH_IOU_HIGH);
      for (const auto &m : high_match.matches) update_track(tracks[m.first], detections[m.second]);

      MatchResult low_match = match_by_iou(tracks, detections, high_match.unmatched_tracks, low_dets, BT_MATCH_IOU_LOW);
      for (const auto &m : low_match.matches) update_track(tracks[m.first], detections[m.second]);

      for (int di : high_match.unmatched_detections) {
        const RawDetection &det = detections[di];
        Track track;
        track.id = next_track_id++;
        init_kalman(track, det.box);
        track.history.push_back({det.cls_id, det.confidence, true});
        track.raw_class_counts[det.cls_id] = 1;
        track.raw_conf_sums[det.cls_id] = det.confidence;
        track.matched_this_frame = true;
        track.age_frames = 1;
        tracks.push_back(std::move(track));
        candidate_tracks_created++;
      }

      for (auto &track : tracks) {
        TrackEval e = evaluate_track(track, track_frames, track_ratio, consecutive_ratio, avg_conf_threshold);
        if (e.full_window) {
          track.ever_full_window = true;
          track.ever_presence_ok = track.ever_presence_ok || e.presence_ok;
          track.ever_consecutive_ok = track.ever_consecutive_ok || e.consecutive_ok;
          track.ever_confidence_ok = track.ever_confidence_ok || e.confidence_ok;
        }
        if (e.confirmed) {  // Hysteresis: keep old class until another class passes all conditions.
          track.confirmed_avg_confidence = e.avg_confidence;
          set_confirmed_class(track, e.best_class);  
        }
      }

      for (auto &track : tracks) {
        if (!track.secondary_pending || track.secondary_checked || !track.matched_this_frame ||
          track.confirmed_class < 0) continue;
        if (coco_resnet_validation_mode(track.confirmed_class) == RESNET_NONE) continue;
        if (frame_number < track.secondary_next_retry_frame) continue;
        ResNetValidationMode validation_mode = coco_resnet_validation_mode(track.confirmed_class);

        int x1 = std::max(0, std::min(static_cast<int>(std::lround(track.box.x)), frame.cols - 1));
        int y1 = std::max(0, std::min(static_cast<int>(std::lround(track.box.y)), frame.rows - 1));
        int x2 = std::max(0, std::min(static_cast<int>(std::lround(track.box.x + track.box.width - 1)), frame.cols - 1));
        int y2 = std::max(0, std::min(static_cast<int>(std::lround(track.box.y + track.box.height - 1)), frame.rows - 1));
        if (x2 <= x1 || y2 <= y1) continue;

        // Rockchip image_utils uses RGA only when source and destination widths are 16-pixel aligned on RK3576.
        // Expand the crop horizontally to the next multiple of 16 without cutting the detected object.
        int crop_w = x2 - x1 + 1;
        int aligned_w = (crop_w + 15) & ~15;
        if (aligned_w <= frame.cols) {
          int extra = aligned_w - crop_w;
          x1 -= extra / 2;
          x2 += extra - extra / 2;
          if (x1 < 0) { x2 -= x1; x1 = 0; }
          if (x2 >= frame.cols) { int shift = x2 - frame.cols + 1; x1 -= shift; x2 -= shift; }
        }

        cv::Mat crop_bgr = frame(cv::Rect(x1, y1, x2 - x1 + 1, y2 - y1 + 1)).clone();
        cv::Mat crop_rgb;
        cv::cvtColor(crop_bgr, crop_rgb, cv::COLOR_BGR2RGB);

        image_buffer_t crop_image;
        memset(&crop_image, 0, sizeof(crop_image));
        crop_image.width = crop_rgb.cols;
        crop_image.height = crop_rgb.rows;
        crop_image.width_stride = crop_rgb.cols;
        crop_image.height_stride = crop_rgb.rows;
        crop_image.format = IMAGE_FORMAT_RGB888;
        crop_image.virt_addr = crop_rgb.data;
        crop_image.size = crop_rgb.total() * crop_rgb.elemSize();

        secondary_classifier_result_t sr;
        if (validation_mode == RESNET_STRICT) secondary_strict_checks++;
        else if (validation_mode == RESNET_ADVISORY) secondary_advisory_checks++;
        int secondary_ret = classify_secondary(&secondary_ctx, &crop_image, &sr);
        secondary_checks++;

        if (secondary_ret != 0) {
          secondary_failures++;
          if (validation_mode == RESNET_ADVISORY && track.secondary_retry_count >= ADVISORY_MAX_RETRIES) {
            secondary_advisory_fallbacks++;
            set_final_class(track, track.confirmed_class);
            track.secondary_checked = true;
            track.secondary_pending = false;
            track.secondary_next_retry_frame = 0;
            printf("Secondary check T%d failed: ret=%d [ADVISORY FALLBACK] -> FINAL=%s\n",
              track.id, secondary_ret, coco_cls_to_name(track.final_class));
          } else {
            track.secondary_retry_count++;
            track.secondary_checked = false;
            track.secondary_pending = true;
            track.secondary_next_retry_frame = frame_number + SECONDARY_RETRY_FRAMES;
            printf("Secondary check T%d failed: ret=%d [PENDING, retry %d%s in %llu frames]\n",
              track.id, secondary_ret, track.secondary_retry_count,
              validation_mode == RESNET_ADVISORY ? "/2" : "", SECONDARY_RETRY_FRAMES);
          }
          continue;
        }

        track.secondary_class = sr.coco_class;
        track.secondary_score = sr.score;
        track.secondary_mapped_mass = sr.mapped_mass;

        if (sr.coco_class < 0) {
          secondary_unmapped++;
          if (validation_mode == RESNET_ADVISORY && track.secondary_retry_count >= ADVISORY_MAX_RETRIES) {
            secondary_advisory_fallbacks++;
            set_final_class(track, track.confirmed_class);
            track.secondary_checked = true;
            track.secondary_pending = false;
            track.secondary_next_retry_frame = 0;
            printf("Secondary check T%d: YOLO=%s %.2f%% -> ResNet=UNMAPPED [ADVISORY FALLBACK] -> FINAL=%s\n",
              track.id, coco_cls_to_name(track.confirmed_class), track.confirmed_avg_confidence * 100.0f,
              coco_cls_to_name(track.final_class));
          } else {
            track.secondary_retry_count++;
            track.secondary_checked = false;
            track.secondary_pending = true;
            track.secondary_next_retry_frame = frame_number + SECONDARY_RETRY_FRAMES;
            printf("Secondary check T%d: YOLO=%s %.2f%% -> ResNet=UNMAPPED [PENDING, retry %d%s in %llu frames]\n",
              track.id, coco_cls_to_name(track.confirmed_class), track.confirmed_avg_confidence * 100.0f,
              track.secondary_retry_count, validation_mode == RESNET_ADVISORY ? "/2" : "",
              SECONDARY_RETRY_FRAMES);
          }
        } else if (sr.coco_class == track.confirmed_class) {
          secondary_agreements++;
          set_final_class(track, track.confirmed_class);
          track.secondary_checked = true;
          track.secondary_pending = false;
          track.secondary_next_retry_frame = 0;
          track.secondary_retry_count = 0;
          printf("Secondary check T%d: YOLO=%s %.2f%% -> ResNet=%s %.2f%% [AGREE] -> FINAL=%s\n", track.id,
            coco_cls_to_name(track.confirmed_class), track.confirmed_avg_confidence * 100.0f,
            coco_cls_to_name(sr.coco_class), sr.score * 100.0f, coco_cls_to_name(track.final_class));
        } else {
          secondary_disagreements++;
          secondary_disagreement_pairs[{track.confirmed_class, sr.coco_class}]++;

          if (sr.score >= resnet_override_threshold) {
            secondary_overrides++;
            secondary_override_pairs[{track.confirmed_class, sr.coco_class}]++;
            set_final_class(track, sr.coco_class);
            track.secondary_checked = true;
            track.secondary_pending = false;
            track.secondary_next_retry_frame = 0;
            track.secondary_retry_count = 0;
            printf("Secondary check T%d: YOLO=%s %.2f%% -> ResNet=%s %.2f%% [OVERRIDE] -> FINAL=%s\n", track.id,
              coco_cls_to_name(track.confirmed_class), track.confirmed_avg_confidence * 100.0f,
              coco_cls_to_name(sr.coco_class), sr.score * 100.0f, coco_cls_to_name(track.final_class));
          } else {
            secondary_below_threshold++;
            if (validation_mode == RESNET_ADVISORY && track.secondary_retry_count >= ADVISORY_MAX_RETRIES) {
              secondary_advisory_fallbacks++;
              set_final_class(track, track.confirmed_class);
              track.secondary_checked = true;
              track.secondary_pending = false;
              track.secondary_next_retry_frame = 0;
              printf("Secondary check T%d: YOLO=%s %.2f%% -> ResNet=%s %.2f%% "
                "[BELOW RS %.0f%%, ADVISORY FALLBACK] -> FINAL=%s\n",
                track.id, coco_cls_to_name(track.confirmed_class), track.confirmed_avg_confidence * 100.0f,
                coco_cls_to_name(sr.coco_class), sr.score * 100.0f, resnet_override_threshold * 100.0f,
                coco_cls_to_name(track.final_class));
            } else {
              set_final_class(track, -1);
              track.secondary_retry_count++;
              track.secondary_checked = false;
              track.secondary_pending = true;
              track.secondary_next_retry_frame = frame_number + SECONDARY_RETRY_FRAMES;
              printf("Secondary check T%d: YOLO=%s %.2f%% -> ResNet=%s %.2f%% "
                "[BELOW RS %.0f%%, PENDING, retry %d%s in %llu frames]\n",
                track.id, coco_cls_to_name(track.confirmed_class), track.confirmed_avg_confidence * 100.0f,
                coco_cls_to_name(sr.coco_class), sr.score * 100.0f, resnet_override_threshold * 100.0f,
                track.secondary_retry_count, validation_mode == RESNET_ADVISORY ? "/2" : "",
                SECONDARY_RETRY_FRAMES);
            }
          }
        }
      }

      for (auto it = tracks.begin(); it != tracks.end();) {
        if (it->missed_frames > max_lost_frames) {
          if (it->confirmed_class < 0) count_rejected_track(*it);
          it = tracks.erase(it);
        } else ++it;
      }

      auto frame_end = std::chrono::steady_clock::now();
      double inference_ms = std::chrono::duration<double, std::milli>(inference_end - inference_start).count();
      double frame_ms = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
      double processing_fps = frame_ms > 0.0 ? 1000.0 / frame_ms : 0.0;
      auto now = std::chrono::steady_clock::now();
      perf_samples.push_back({now, inference_ms, processing_fps});

      while (!perf_samples.empty()) {
        double age = std::chrono::duration<double>(now - perf_samples.front().time).count();
        if (age <= PERF_WINDOW_SECONDS) break;
        perf_samples.pop_front();
      }

      double npu_sum = 0.0; double fps_sum = 0.0;
      double npu_min = std::numeric_limits<double>::max(); double npu_max = 0.0;
      double fps_min = std::numeric_limits<double>::max(); double fps_max = 0.0;
      for (const auto &s : perf_samples) {
        npu_sum += s.npu_ms; fps_sum += s.fps;
        npu_min = std::min(npu_min, s.npu_ms); npu_max = std::max(npu_max, s.npu_ms);
        fps_min = std::min(fps_min, s.fps); fps_max = std::max(fps_max, s.fps);
      }
      double npu_avg = perf_samples.empty() ? 0.0 : npu_sum / perf_samples.size();
      double fps_avg = perf_samples.empty() ? 0.0 : fps_sum / perf_samples.size();

      double scale = std::min(static_cast<double>(screen_width) / frame.cols, static_cast<double>(screen_height) / frame.rows);
      int dst_w = static_cast<int>(frame.cols * scale); int dst_h = static_cast<int>(frame.rows * scale);
      cv::Mat display(screen_height, screen_width, CV_8UC3, cv::Scalar(0, 0, 0));
      cv::Mat resized;
      cv::resize(frame, resized, cv::Size(dst_w, dst_h));
      int x = (screen_width - dst_w) / 2; int y = (screen_height - dst_h) / 2;
      resized.copyTo(display(cv::Rect(x, y, dst_w, dst_h)));

      // Draw FINAL objects in display coordinates so source resolution does not affect UI size.
      const int box_thickness = ui_px(3);
      const double label_font_scale = 0.90 * ui_scale;
      const int label_thickness = ui_px(2);
      const int label_pad_x = ui_px(7);
      const int label_pad_y = ui_px(5);

      for (const auto &track : tracks) {
        if (!track.matched_this_frame || track.final_class < 0) continue;

        int sx1 = std::max(0, std::min(static_cast<int>(std::lround(track.box.x)), frame.cols - 1));
        int sy1 = std::max(0, std::min(static_cast<int>(std::lround(track.box.y)), frame.rows - 1));
        int sx2 = std::max(0,
          std::min(static_cast<int>(std::lround(track.box.x + track.box.width - 1)), frame.cols - 1));
        int sy2 = std::max(0,
          std::min(static_cast<int>(std::lround(track.box.y + track.box.height - 1)), frame.rows - 1));
        if (sx2 <= sx1 || sy2 <= sy1) continue;

        int dx1 = x + static_cast<int>(std::lround(sx1 * scale));
        int dy1 = y + static_cast<int>(std::lround(sy1 * scale));
        int dx2 = x + static_cast<int>(std::lround(sx2 * scale));
        int dy2 = y + static_cast<int>(std::lround(sy2 * scale));
        dx1 = std::max(x, std::min(dx1, x + dst_w - 1));
        dy1 = std::max(y, std::min(dy1, y + dst_h - 1));
        dx2 = std::max(x, std::min(dx2, x + dst_w - 1));
        dy2 = std::max(y, std::min(dy2, y + dst_h - 1));
        if (dx2 <= dx1 || dy2 <= dy1) continue;

        const TrackColor &colors = track_color(track.id);
        cv::rectangle(display, cv::Point(dx1, dy1), cv::Point(dx2, dy2), colors.bg, box_thickness);

        char label[128];
        snprintf(label, sizeof(label), "%s [T%d]", coco_cls_to_name(track.final_class), track.id);

        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, label_font_scale,
          label_thickness, &baseline);
        int text_y = std::max(dy1, y + text_size.height + label_pad_y * 2);
        int label_right = std::min(screen_width - 1, dx1 + text_size.width + label_pad_x * 2);
        int label_top = std::max(0, text_y - text_size.height - label_pad_y * 2);

        cv::rectangle(display, cv::Point(dx1, label_top), cv::Point(label_right, text_y + baseline),
          colors.bg, cv::FILLED);
        cv::putText(display, label, cv::Point(dx1 + label_pad_x, text_y - label_pad_y),
          cv::FONT_HERSHEY_SIMPLEX, label_font_scale, colors.text, label_thickness, cv::LINE_AA);
      }

      // Show Info
      if (show_info) {
        char info_line[256];
        int baseline = 0; double font_scale = 0.55 * ui_scale; int thickness = ui_px(1);
        int font_face = cv::FONT_HERSHEY_SIMPLEX;
        snprintf(info_line, sizeof(info_line), "FPS %.1f (%.1f-%.1f)  INF %.1f (%.1f-%.1f)ms",
          fps_avg, fps_min, fps_max, npu_avg, npu_min, npu_max);
        cv::Size text_size = cv::getTextSize(info_line, font_face, font_scale, thickness, &baseline);
        const int pad_x = ui_px(7), pad_y = ui_px(5), margin_x = ui_px(10), panel_y = ui_px(8);
        int box_w = text_size.width + pad_x * 2;
        int box_h = text_size.height + baseline + pad_y * 2;
        int box_x = screen_width - margin_x - box_w;
        cv::Rect roi_rect(box_x, panel_y, box_w, box_h);
        cv::Mat roi = display(roi_rect); cv::Mat overlay = roi.clone();
        overlay.setTo(cv::Scalar(0, 0, 0)); double alpha = 0.6;
        cv::addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, roi);
        int text_x = screen_width - margin_x - pad_x - text_size.width;
        int text_y = panel_y + pad_y + text_size.height;
        cv::putText(display, info_line, cv::Point(text_x, text_y), font_face, font_scale,
          cv::Scalar(0, 255, 0), thickness, cv::LINE_AA);
      }

      // Show Statistics
      if (show_statistics) {
        int active_unconfirmed = 0;
        int active_pending = 0;
        for (const auto &track : tracks) {
          if (track.confirmed_class < 0) active_unconfirmed++;
          else if (track.final_class < 0) active_pending++;
        }
        const int panel_x = ui_px(10), panel_y = ui_px(8), panel_w = ui_px(250), line_h = ui_px(23);
        int lines = 16 + static_cast<int>(final_objects.size());
        int panel_h = std::min(screen_height - panel_y - ui_px(5), lines * line_h + ui_px(15));
        cv::Rect roi_rect(panel_x, panel_y, panel_w, panel_h);
        cv::Mat roi = display(roi_rect); cv::Mat overlay = roi.clone();
        overlay.setTo(cv::Scalar(0, 0, 0)); double alpha = 0.6;
        cv::addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, roi);
        int ty = panel_y + ui_px(25);
        auto put_line = [&](const std::string &txt) {
          if (ty + ui_px(5) > panel_y + panel_h) return;
          cv::putText(display, txt, cv::Point(panel_x + ui_px(10), ty), cv::FONT_HERSHEY_SIMPLEX,
            0.48 * ui_scale, cv::Scalar(255, 255, 255), ui_px(1), cv::LINE_AA);
          ty += line_h;
        };
        char line[256];
        snprintf(line, sizeof(line), "Confirmed tracks: %llu", confirmed_tracks_total); put_line(line);
        snprintf(line, sizeof(line), "Candidates created: %llu", candidate_tracks_created); put_line(line);
        snprintf(line, sizeof(line), "Active unconfirmed: %d", active_unconfirmed); put_line(line);
        snprintf(line, sizeof(line), "Active pending: %d", active_pending); put_line(line);
        snprintf(line, sizeof(line), "Rejected candidates: %llu", rejected_tracks_total); put_line(line);
        snprintf(line, sizeof(line), "Reclassifications: %llu", reclassifications); put_line(line);
        snprintf(line, sizeof(line), "ResNet checks: %llu", secondary_checks); put_line(line);
        snprintf(line, sizeof(line), "  strict: %llu", secondary_strict_checks); put_line(line);
        snprintf(line, sizeof(line), "  advisory: %llu", secondary_advisory_checks); put_line(line);
        snprintf(line, sizeof(line), "  agree: %llu", secondary_agreements); put_line(line);
        snprintf(line, sizeof(line), "  disagree: %llu", secondary_disagreements); put_line(line);
        snprintf(line, sizeof(line), "  overrides: %llu", secondary_overrides); put_line(line);
        snprintf(line, sizeof(line), "  adv. fallbacks: %llu", secondary_advisory_fallbacks); put_line(line);
        snprintf(line, sizeof(line), "  unmapped/fail: %llu/%llu", secondary_unmapped, secondary_failures); put_line(line);
        put_line("Final objects:");
        for (const auto &entry : final_objects) {
          snprintf(line, sizeof(line), "  %-14s %llu", coco_cls_to_name(entry.first), entry.second); put_line(line);
        }
      }

      // Playback delay OSD: visible for 5 seconds after +/- is pressed.
      if (std::chrono::steady_clock::now() < playback_delay_osd_until) {
        char delay_text[64];
        snprintf(delay_text, sizeof(delay_text), "Playback delay: %d ms/frame", playback_delay_ms);

        const int font_face = cv::FONT_HERSHEY_SIMPLEX;
        const double font_scale = 0.55 * ui_scale;
        const int thickness = ui_px(1);
        const int pad_x = ui_px(7), pad_y = ui_px(5), margin_x = ui_px(10), margin_y = ui_px(10);
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(delay_text, font_face, font_scale, thickness, &baseline);
        int box_w = text_size.width + pad_x * 2;
        int box_h = text_size.height + baseline + pad_y * 2;
        int box_x = screen_width - margin_x - box_w;
        int box_y = screen_height - margin_y - box_h;

        cv::Rect roi_rect(box_x, box_y, box_w, box_h);
        cv::Mat roi = display(roi_rect); cv::Mat overlay = roi.clone();
        overlay.setTo(cv::Scalar(0, 0, 0)); double alpha = 0.6;
        cv::addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, roi);

        int text_x = screen_width - margin_x - pad_x - text_size.width;
        int text_y = box_y + pad_y + text_size.height;
        cv::putText(display, delay_text, cv::Point(text_x, text_y), font_face, font_scale,
          cv::Scalar(0, 255, 0), thickness, cv::LINE_AA);
      }

      cv::imshow("YOLOv8 - RK3576 NPU", display);

      int key = cv::waitKey(playback_delay_ms + 1);
      if (key == 's' || key == 'S') show_statistics = !show_statistics;
      if (key == 'i' || key == 'I') show_info = !show_info;
      if (key == '+' || key == '=') {
        playback_delay_ms = std::max(0, playback_delay_ms - PLAYBACK_DELAY_STEP_MS);
        playback_delay_osd_until = std::chrono::steady_clock::now() +
          std::chrono::milliseconds(static_cast<int>(PLAYBACK_DELAY_OSD_SECONDS * 1000.0));
      }
      if (key == '-' || key == '_') {
        playback_delay_ms = std::min(PLAYBACK_DELAY_MAX_MS, playback_delay_ms + PLAYBACK_DELAY_STEP_MS);
        playback_delay_osd_until = std::chrono::steady_clock::now() +
          std::chrono::milliseconds(static_cast<int>(PLAYBACK_DELAY_OSD_SECONDS * 1000.0));
      }
      if (key == 27 || key == 'q' || key == 'Q') break;
    }

    for (const auto &track : tracks) if (track.confirmed_class < 0) count_rejected_track(track);

    printf("\n========== Tracking statistics ==========\n");
    printf("Window                  : %d ms (%d frames)\n", track_ms, track_frames);
    printf("Validation              : TR %.0f%% (%d/%d), TC %.0f%% (%d), AC %.0f%%, CT %.0f%%, HT %.0f%%\n",
      track_ratio * 100, required_presence, track_frames,
      consecutive_ratio * 100, required_consecutive, avg_conf_threshold * 100, conf_thresh * 100, high_thresh * 100);
    printf("Candidate tracks created: %llu\n", candidate_tracks_created);
    printf("Confirmed tracks        : %llu\n", confirmed_tracks_total);
    printf("Rejected candidates     : %llu\n", rejected_tracks_total);
    printf("Reclassifications       : %llu\n", reclassifications);
    if (!confirmed_objects.empty()) {
      printf("\nTemporally confirmed objects:\n");
      for (const auto &entry : confirmed_objects) printf("  %-14s : %llu\n", coco_cls_to_name(entry.first), entry.second);
    }
    if (!rejected_classes.empty()) {
      printf("\nRejected candidates (best raw class):\n");
      for (const auto &entry : rejected_classes) printf("  %-14s : %llu\n", coco_cls_to_name(entry.first), entry.second);
    }
    if (!final_objects.empty()) {
      printf("\nFinal objects:\n");
      for (const auto &entry : final_objects) printf("  %-14s : %llu\n", coco_cls_to_name(entry.first), entry.second);
    }
    printf("\nReject reasons (can overlap):\n");
    printf("  too short for full window : %llu\n", rejected_short_lived);
    printf("  never enough presence     : %llu\n", rejected_low_presence);
    printf("  never enough consecutive  : %llu\n", rejected_low_consecutive);
    printf("  never enough avg confidence: %llu\n", rejected_low_confidence);

    printf("\nSecondary classifier:\n");
    printf("  checks                    : %llu\n", secondary_checks);
    printf("    strict                  : %llu\n", secondary_strict_checks);
    printf("    advisory                : %llu\n", secondary_advisory_checks);
    printf("  agreement                 : %llu\n", secondary_agreements);
    printf("  disagreement              : %llu\n", secondary_disagreements);
    printf("  unmapped                  : %llu\n", secondary_unmapped);
    printf("  failures                  : %llu\n", secondary_failures);
    printf("  overrides                 : %llu\n", secondary_overrides);
    printf("  advisory fallbacks        : %llu\n", secondary_advisory_fallbacks);
    printf("  disagree below RS         : %llu\n", secondary_below_threshold);
    if (!secondary_disagreement_pairs.empty()) {
      printf("  disagreements:\n");
      for (const auto &entry : secondary_disagreement_pairs)
        printf("    %-14s -> %-14s : %llu\n", coco_cls_to_name(entry.first.first),
          coco_cls_to_name(entry.first.second), entry.second);
    }
    if (!secondary_override_pairs.empty()) {
      printf("  overrides applied (RS %.0f%%):\n", resnet_override_threshold * 100.0f);
      for (const auto &entry : secondary_override_pairs)
        printf("    %-14s -> %-14s : %llu\n", coco_cls_to_name(entry.first.first),
          coco_cls_to_name(entry.first.second), entry.second);
    }
    printf("=========================================\n\n");

    cap.release();
    cv::destroyAllWindows();
    release_secondary_classifier(&secondary_ctx);
    if (use_zero_copy) ret = release_yolov8_model_zc(&rknn_app_ctx); else ret = release_yolov8_model(&rknn_app_ctx);
    if (ret != 0) printf("release_yolov8_model failed! ret=%d\n", ret);
    deinit_post_process();
    return 0;
}
