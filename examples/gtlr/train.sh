#!/usr/bin/env bash
# Train GTLR-GS from artifacts produced by preprocess.sh.
#
# Required:
#   DATA=/path/to/colmap_dataset ./gtlr/train.sh
# Extra arguments are forwarded to simple_trainer_gtlr, for example:
#   DATA=... MAX_STEPS=4000 ./gtlr/train.sh --save_steps 4000
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXAMPLES_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

if [[ -z "${DATA:-}" ]]; then
    echo "DATA is required (COLMAP dataset containing images and sparse/0)" >&2
    exit 2
fi

WORK_DIR=${WORK_DIR:-"$DATA/gtlr"}
INIT_PLY=${INIT_PLY:-"$WORK_DIR/init.ply"}
DEPTH_DIR=${DEPTH_DIR:-"$WORK_DIR/depth_maps"}
RESULT_DIR=${RESULT_DIR:-"$WORK_DIR/results"}
FACTOR=${FACTOR:-4}
MAX_STEPS=${MAX_STEPS:-30000}
DEPTH_START_ITER=${DEPTH_START_ITER:-3000}
DEPTH_LAMBDA=${DEPTH_LAMBDA:-1.0}
NORMAL_LAMBDA=${NORMAL_LAMBDA:-0.05}
CURVATURE_REF_MAX=${CURVATURE_REF_MAX:-50000}
CUDA_DEVICE=${CUDA_DEVICE:-0}

if [[ ! -f "$INIT_PLY" || ! -f "$INIT_PLY.json" ]]; then
    echo "Missing sampled initialization or sidecar: $INIT_PLY[.json]" >&2
    echo "Run examples/gtlr/preprocess.sh first." >&2
    exit 2
fi
if [[ ! -f "$DEPTH_DIR/manifest.json" ]]; then
    echo "Missing depth manifest: $DEPTH_DIR/manifest.json" >&2
    echo "Run examples/gtlr/preprocess.sh first." >&2
    exit 2
fi

mkdir -p "$RESULT_DIR"
cd "$EXAMPLES_DIR"

echo "GTLR training in raw metric LiDAR/COLMAP coordinates"
echo "  data:   $DATA"
echo "  init:   $INIT_PLY"
echo "  depth:  $DEPTH_DIR"
echo "  result: $RESULT_DIR"
echo "  factor: $FACTOR"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python -u -m gtlr.simple_trainer_gtlr \
    --data_dir "$DATA" \
    --data_factor "$FACTOR" \
    --init_ply "$INIT_PLY" \
    --depth_dir "$DEPTH_DIR" \
    --result_dir "$RESULT_DIR" \
    --max_steps "$MAX_STEPS" \
    --depth_start_iter "$DEPTH_START_ITER" \
    --depth_lambda "$DEPTH_LAMBDA" \
    --normal_lambda "$NORMAL_LAMBDA" \
    --strategy.curvature-ref-max "$CURVATURE_REF_MAX" \
    "$@"
