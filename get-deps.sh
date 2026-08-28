#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_DIR="$ROOT_DIR/.rknn_model_zoo_tmp"

if [[ -d "$ROOT_DIR/3rdparty" && -d "$ROOT_DIR/utils" ]]; then
  echo "[OK] Dependencies already present."
  exit 0
fi

echo "==> Downloading RKNN Model Zoo v2.3.0 dependencies..."

rm -rf "$TMP_DIR"

git clone \
  --branch v2.3.0 \
  --depth 1 \
  --filter=blob:none \
  --sparse \
  https://github.com/airockchip/rknn_model_zoo.git \
  "$TMP_DIR"

cd "$TMP_DIR"
git sparse-checkout set 3rdparty utils

cp -a "$TMP_DIR/3rdparty" "$ROOT_DIR/"
cp -a "$TMP_DIR/utils" "$ROOT_DIR/"

cd "$ROOT_DIR"
rm -rf "$TMP_DIR"

echo "[OK] Dependencies installed."
