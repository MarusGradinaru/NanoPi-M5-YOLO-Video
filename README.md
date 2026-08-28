# YOLOv8 Video Object Detection for RK3576

A real-time object detection and tracking application for Rockchip RK3576 devices, built around RKNN, YOLOv8 and OpenCV.

The project was developed and tested on a NanoPi M5 and is intended for video files or camera-style workloads where stable object detection is more useful than raw frame-by-frame predictions.

Instead of displaying every YOLO detection immediately, the application combines object detection, tracking and temporal validation so that short-lived or unstable detections are filtered before they reach the final output.

## Main features

- YOLOv8 object detection accelerated by the RK3576 NPU
- Zero-copy inference enabled by default
- ByteTrack-style multi-object tracking with persistent track IDs
- Temporal validation to reduce unstable or one-frame detections
- Optional secondary ResNet50 validation for selected object classes
- Three secondary validation modes: `NONE`, `ADVISORY` and `STRICT`
- Automatic UI scaling for different video resolutions
- Fullscreen OpenCV display
- Toggleable performance and statistics overlays
- Adjustable playback delay for easier visual inspection
- Command-line configurable detection and validation thresholds

## How it works

Each frame passes through the following stages:

```text
Video frame
    ↓
YOLOv8 detection
    ↓
Object tracking
    ↓
Temporal validation
    ↓
Optional ResNet50 validation
    ↓
Final object
    ↓
Display
```

YOLO provides the raw detections. The tracker links detections between frames and assigns persistent IDs such as `T1`, `T2`, etc.

A track must remain sufficiently stable over time before it is considered confirmed. Depending on the object class, a secondary ResNet50 classifier may also be used as an additional semantic check.

Only final validated objects are displayed.

## Target platform

This project is intended for:

- NanoPi M5 (Rockchip RK3576) hardware
- FriendlyElec Ubuntu Noble GNOME Desktop OS
- RKNN Runtime / RKNPU2
- OpenCV
- CMake
- GCC / G++

It was developed and tested on a NanoPi M5.

The supplied RKNN models are built for RK3576.

## Clone and prepare

Clone the repository and enter its directory:

```bash
git clone https://github.com/MarusGradinaru/NanoPi-M5-YOLO-Video.git
cd NanoPi-M5-YOLO-Video
```

Download the required Rockchip helper directories:

```bash
./get-deps.sh
```

The script downloads only the required `3rdparty/` and `utils/` directories from `rknn_model_zoo` v2.3.0 using Git sparse checkout.

You only need to run this once unless you remove those directories.

## Build

For the first build:

```bash
./build.sh fresh
```

This removes any existing build artifacts, creates a clean `build/` directory, runs CMake and compiles the project.

For normal rebuilds:

```bash
./build.sh
```

If the `build/` directory does not exist, `build.sh` automatically switches to a fresh build.

After a successful build, the executable is placed in the project root:

```text
./yolov8-video
```

The application expects its models in the `model/` directory next to the executable.

## Running

Basic usage:

```bash
./yolov8-video <video_file>
```

Example:

```bash
./yolov8-video test.mp4
```

Zero-copy is enabled by default.

Default models:

```text
YOLO   : model/yolov8s_rk3576_i8.rknn
ResNet : model/resnet50-v2-7-i8.rknn
```

## Command-line syntax

```text
./yolov8-video <video_file> [yolo:<model.rknn>] [rnet:<model.rknn>] [no-zc]
               [ct:1-100] [ht:1-100] [tw:200-5000]
               [tr:1-100] [tc:1-100] [ac:1-100] [rs:1-100]
```

| Option | Description | Default |
|---|---|---:|
| `yolo:<file>` | YOLO RKNN model filename | `yolov8s_rk3576_i8.rknn` |
| `rnet:<file>` | ResNet50 RKNN model filename | `resnet50-v2-7-i8.rknn` |
| `no-zc` | Disable zero-copy inference | zero-copy enabled |
| `ct:<1-100>` | Minimum raw confidence accepted by the tracker | `40` |
| `ht:<1-100>` | High-confidence threshold used to create tracks | `55` |
| `tw:<200-5000>` | Tracking / validation window in milliseconds | `1000` |
| `tr:<1-100>` | Required presence ratio inside the tracking window | `80` |
| `tc:<1-100>` | Required consecutive detection ratio | `30` |
| `ac:<1-100>` | Required average confidence for temporal confirmation | `65` |
| `rs:<1-100>` | Minimum ResNet score required for a class override | `80` |

Example:

```bash
./yolov8-video test.mp4 yolo:yolov8s_rk3576_i8.rknn ct:45 tw:1200
```

Run without zero-copy:

```bash
./yolov8-video test.mp4 no-zc
```

## Keyboard controls

| Key | Action |
|---|---|
| `I` | Toggle performance information |
| `S` | Toggle tracking / detection statistics |
| `+` | Reduce playback delay / play faster |
| `-` | Increase playback delay / play slower |
| `Q` or `Esc` | Quit |

The playback controls only add or remove an intentional delay after each processed frame. They do not make the detection pipeline run faster than the hardware is capable of.

## On-screen information

Final detections are shown using the object class and persistent track ID:

```text
dog [T10]
person [T3]
chair [T17]
```

Press `I` to show processing FPS and inference time.

Press `S` to show live tracking and validation statistics.

## Detection behavior

The application intentionally favors stable detections over maximum recall.

A real object may be filtered if it is detected too inconsistently or with insufficient confidence. This is expected behavior and helps prevent unstable or short-lived detections from reaching the final output.

The secondary ResNet classifier is not used equally for every COCO class. Classes use one of three modes:

- `NONE` — temporal YOLO validation is sufficient
- `ADVISORY` — ResNet may confirm or override, but weak disagreement eventually falls back to YOLO
- `STRICT` — ResNet validation is mandatory

The class policy is defined in `imagenet_to_coco80.h` and can be adjusted if additional testing shows that a class should use a different validation mode.

## External dependencies and licenses

This project uses code, helper utilities, models and/or components from other open-source projects, including Rockchip's RKNN ecosystem and `rknn_model_zoo`.

The helper directories downloaded by `get-deps.sh` come from:

```text
Rockchip rknn_model_zoo v2.3.0
```

All third-party components retain their original licenses, copyright notices and distribution terms.

This repository does not replace, override or relicense third-party components. Anyone redistributing this project or bundled third-party files should review and comply with the licenses that apply to those components.

## Acknowledgements

Thanks to the Rockchip RKNN ecosystem and the authors and maintainers of the software and models used as building blocks for this project.
