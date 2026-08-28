#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_DIR="$ROOT_DIR/.rknn_model_zoo_tmp"

need_3rdparty=false
need_utils=false

[[ -d "$ROOT_DIR/3rdparty" ]] || need_3rdparty=true
[[ -d "$ROOT_DIR/utils" ]] || need_utils=true

if ! $need_3rdparty && ! $need_utils; then
  echo "[-OK-] Dependencies already present."
  exit 0
fi

echo "==> Downloading dependencies from RKNN Model Zoo v2.3.0..."

rm -rf "$TMP_DIR"

git clone \
  --branch v2.3.0 \
  --depth 1 \
  --filter=blob:none \
  --sparse \
  https://github.com/airockchip/rknn_model_zoo.git \
  "$TMP_DIR"

cd "$TMP_DIR"
if $need_3rdparty && $need_utils; then
  git sparse-checkout set 3rdparty utils
elif $need_3rdparty; then
  git sparse-checkout set 3rdparty
else
  git sparse-checkout set utils
fi
cd "$ROOT_DIR"

echo "==> Installing dependencies..."

if $need_3rdparty; then
  mv "$TMP_DIR/3rdparty" "$ROOT_DIR/"
  echo "  [OK] Installed: 3rdparty"
fi

if $need_utils; then
  mv "$TMP_DIR/utils" "$ROOT_DIR/"
  echo "  [OK] Installed: utils"
fi

rm -rf "$TMP_DIR"

if $need_utils; then
  echo "==> Applying local patches..."
fi

if $need_utils; then
  if ! patch -d "$ROOT_DIR" -p1 < "$ROOT_DIR/patches/rknn_model_zoo-v2.3.0-silent-image-utils.patch"; then
    echo "[ERROR] Failed to patch 'utils' dependency !"
    rm -rf "$ROOT_DIR/utils"
    exit 1
  fi
fi

echo "[-OK-] All dependencies are installed."
