#!/usr/bin/env bash
# GTLR-GS preprocessing in the raw metric LiDAR/COLMAP frame.
#
# Required:
#   DATA=/path/to/colmap_dataset ./gtlr/preprocess.sh
# Optional environment variables are documented in README.md next to this file.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXAMPLES_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

if [[ -z "${DATA:-}" ]]; then
    echo "DATA is required (COLMAP dataset containing images and sparse/0)" >&2
    exit 2
fi

SRC_PLY=${SRC_PLY:-"$DATA/sparse/0/extracted_all.ply"}
WORK_DIR=${WORK_DIR:-"$DATA/gtlr"}
INIT_PLY=${INIT_PLY:-"$WORK_DIR/init.ply"}
DEPTH_DIR=${DEPTH_DIR:-"$WORK_DIR/depth_maps"}
VALIDATE_DIR=${VALIDATE_DIR:-"$WORK_DIR/validate_preprocess"}
NUM_SAMPLES=${NUM_SAMPLES:-1500000}
FACTOR=${FACTOR:-4}
ALLOW_UPSAMPLE=${ALLOW_UPSAMPLE:-0}
BASE_RATIO=${BASE_RATIO:-0}
KNN_DEVICE=${KNN_DEVICE:-auto}
KNN_REF_MAX=${KNN_REF_MAX:-50000}
KNN_CHUNK_SIZE=${KNN_CHUNK_SIZE:-1024}
DEPTH_DEVICE=${DEPTH_DEVICE:-auto}
DEPTH_CHUNK_SIZE=${DEPTH_CHUNK_SIZE:-1000000}
MIN_DEPTH=${MIN_DEPTH:-0}
MAX_DEPTH=${MAX_DEPTH:-inf}
VIS_EVERY=${VIS_EVERY:-100}
RUN_SAMPLING_VALIDATION=${RUN_SAMPLING_VALIDATION:-1}

if [[ ! -d "$DATA/images" || ! -d "$DATA/sparse/0" ]]; then
    echo "Invalid DATA=$DATA: expected images/ and sparse/0/" >&2
    exit 2
fi
if [[ ! -f "$SRC_PLY" ]]; then
    echo "LiDAR PLY does not exist: $SRC_PLY" >&2
    exit 2
fi
mkdir -p "$WORK_DIR" "$VALIDATE_DIR"
cd "$EXAMPLES_DIR"

sample_cmd=(
    python -u -m gtlr.sample_points
    --input_ply "$SRC_PLY"
    --output_ply "$INIT_PLY"
    --num_samples "$NUM_SAMPLES"
    --base_ratio "$BASE_RATIO"
    --device "$KNN_DEVICE"
    --ref_max "$KNN_REF_MAX"
    --chunk_size "$KNN_CHUNK_SIZE"
)
if [[ "$ALLOW_UPSAMPLE" == "1" ]]; then
    sample_cmd+=(--allow_upsample)
fi

echo "[1/3] GTLR geometry-texture allocation: $SRC_PLY -> $INIT_PLY"
"${sample_cmd[@]}"

echo "[2/3] GTLR LiDAR camera-z projection: $DEPTH_DIR"
python -u -m gtlr.project_depth \
    --data_dir "$DATA" \
    --ply "$SRC_PLY" \
    --output_dir "$DEPTH_DIR" \
    --factor "$FACTOR" \
    --device "$DEPTH_DEVICE" \
    --chunk_size "$DEPTH_CHUNK_SIZE" \
    --min_depth "$MIN_DEPTH" \
    --max_depth "$MAX_DEPTH" \
    --vis_every "$VIS_EVERY"

if [[ "$RUN_SAMPLING_VALIDATION" == "1" ]]; then
    echo "[3/3] Validate allocation enrichment"
    python -u -m gtlr.validate \
        --data_dir "$DATA" \
        --full_ply "$SRC_PLY" \
        --sampled_ply "$INIT_PLY" \
        --output_dir "$VALIDATE_DIR"
else
    echo "[3/3] Sampling validation skipped"
fi

echo "Preprocessing complete"
echo "  init:  $INIT_PLY"
echo "  depth: $DEPTH_DIR"
