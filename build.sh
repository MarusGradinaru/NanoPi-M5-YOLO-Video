#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$ROOT_DIR/build"
OUTPUT="$BUILD_DIR/yolov8-video"
DEST="$ROOT_DIR/yolov8-video"

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [fresh]" >&2
  exit 1
fi

missing=()

[[ -d "$ROOT_DIR/3rdparty" ]] || missing+=("3rdparty")
[[ -d "$ROOT_DIR/utils" ]] || missing+=("utils")

if (( ${#missing[@]} > 0 )); then
  echo "[ERROR] Missing dependencies: ${missing[*]}" >&2
  echo "Run: ./get-deps.sh" >&2
  exit 1
fi

MODE="${1:-}"

if [[ -n "$MODE" && "$MODE" != "fresh" ]]; then
  echo "[ERROR] Unknown parameter: $MODE" >&2
  echo "Usage: $0 [fresh]" >&2
  exit 1
fi

if [[ ! -d "$BUILD_DIR" ]]; then
  MODE="fresh"
fi

if [[ "$MODE" == "fresh" ]]; then
  echo "==> Fresh build"
  rm -rf "$BUILD_DIR"
  mkdir -p "$BUILD_DIR"
  cd "$BUILD_DIR"
  cmake ..
else
  echo "==> Incremental build"
  mkdir -p "$BUILD_DIR"
  cd "$BUILD_DIR"
fi

make -j$(nproc)

if [[ ! -f "$OUTPUT" ]]; then
  echo "[ERROR] Executable was not found: $OUTPUT" >&2
  exit 1
fi

mv -f "$OUTPUT" "$DEST"

echo "[-OK-] Output executable: $DEST"
