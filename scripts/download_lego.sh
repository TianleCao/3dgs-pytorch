#!/usr/bin/env bash
# Fetches the NeRF synthetic "lego" scene (Mildenhall et al., 2020) into
# <repo>/data/lego/.
#
# Pulls nerf_synthetic.zip (~1.3 GB, all 8 scenes) from the public Google Drive share, then
# extracts ONLY the lego/ subdirectory via unzip's glob filter -- the other 7
# scenes never touch disk.
#
# Works regardless of where the repo is cloned (paths are resolved relative
# to this script's own location).
#
# Requirements: uv (https://docs.astral.sh/uv/) and unzip. Override the
# upstream link with DRIVE_FILE_ID=<id> if it ever changes.
set -euo pipefail

DRIVE_FILE_ID="${DRIVE_FILE_ID:-1OsiBs2udl32-1CqTXCitmov4NQCYdA9g}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="${REPO_DIR:-$(dirname "$SCRIPT_DIR")}"
DATA_DIR="$REPO_DIR/data"
LEGO_DIR="$DATA_DIR/lego"
WORK_DIR="$(mktemp -d -t nerf-dl-XXXXXX)"
trap 'rm -rf "$WORK_DIR"' EXIT

if [ -d "$LEGO_DIR" ] && [ -n "$(ls -A "$LEGO_DIR" 2>/dev/null)" ]; then
    echo "==> $LEGO_DIR already exists and is non-empty. Nothing to do."
    exit 0
fi

if ! command -v uvx >/dev/null 2>&1; then
    echo "ERROR: 'uvx' not found. Install uv first: https://docs.astral.sh/uv/" >&2
    exit 1
fi
if ! command -v unzip >/dev/null 2>&1; then
    echo "ERROR: 'unzip' not found. Install with: apt-get install -y unzip  (or brew install unzip)" >&2
    exit 1
fi

mkdir -p "$DATA_DIR"

echo "==> Downloading nerf_synthetic.zip (~1.3 GB) via gdown"
uvx --from gdown gdown "$DRIVE_FILE_ID" -O "$WORK_DIR/nerf_synthetic.zip"

echo "==> Extracting only nerf_synthetic/lego/ from the zip"
unzip -q -o "$WORK_DIR/nerf_synthetic.zip" "nerf_synthetic/lego/*" -d "$WORK_DIR/extracted"
mv "$WORK_DIR/extracted/nerf_synthetic/lego" "$LEGO_DIR"

echo
echo "==> Done. Contents of $LEGO_DIR:"
ls -1 "$LEGO_DIR"
echo
echo "    Expected: transforms_train.json, transforms_val.json,"
echo "    transforms_test.json, and train/ val/ test/ image folders."
