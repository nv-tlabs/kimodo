#!/usr/bin/env bash
set -euo pipefail

# End-to-end pipeline for one HUMI ik_recomputed JSON using a distilled
# straight-line Flow-Matching G1 model.
#
# This mirrors scripts/run_humi_ik_to_g1_full_pipeline.sh, but defaults the
# distill config to the DDIM-final-clean FM config and requires DISTILL_CKPT.
#
# Usage:
#   DISTILL_CKPT=outputs/g1_flowmatch_ddim_clean_teacher100_gt_smoke_1gpu/checkpoints/step_00001000.pt \
#   HUMI_SEGMENT_SEED=36 \
#   CUDA_VISIBLE_DEVICES=1 \
#   ROOT_CONSTRAINT_MODE=xyzyaw \
#   KEYFRAME_STEP=20 \
#   DIFFUSION_STEPS=20 \
#   CFG_WEIGHT="2.0 2.0" \
#   bash scripts/run_humi_ik_to_g1_full_pipeline_flowmatch_clean.sh \
#     hf_humi_raw/proposal/ik_recomputed/recording_000.json \
#     proposal_000_ik_36_fm_clean

INPUT="${1:-hf_humi_raw/proposal/ik_recomputed/recording_000.json}"
RUN_NAME="${2:-$(basename "$(dirname "$(dirname "${INPUT}")")")_ik_$(basename "${INPUT}" .json)_fm_clean}"
IK_POSE_SOURCE="${IK_POSE_SOURCE:-realized_target}"
OUT_DIR="${OUT_DIR:-scripts/pipeline_outputs/${RUN_NAME}}"
EXTRACTED_JSON="${OUT_DIR}/ik_${IK_POSE_SOURCE}_pose.json"

export DISTILL_CONFIG="${DISTILL_CONFIG:-kimodo/distillation/configs/flowmatch_g1_ddim_clean_teacher100_gt_smoke_1gpu.yaml}"
DISTILL_CKPT="${DISTILL_CKPT:-}"

if [[ -z "${DISTILL_CKPT}" ]]; then
  echo "Error: DISTILL_CKPT must be set for flowmatch-clean inference."
  exit 1
fi
if [[ ! -f "${DISTILL_CONFIG}" ]]; then
  echo "Error: DISTILL_CONFIG not found: ${DISTILL_CONFIG}"
  exit 1
fi
if [[ ! -f "${DISTILL_CKPT}" ]]; then
  echo "Error: DISTILL_CKPT not found: ${DISTILL_CKPT}"
  exit 1
fi
export DISTILL_CKPT

if [[ "$(basename "$(dirname "${INPUT}")")" == "raw_trajectories" ]]; then
  CANDIDATE_IK_JSON="$(dirname "$(dirname "${INPUT}")")/ik_recomputed/$(basename "${INPUT}")"
  if [[ -f "${CANDIDATE_IK_JSON}" ]]; then
    echo "Input is raw_trajectories; using matching ik_recomputed JSON: ${CANDIDATE_IK_JSON}"
    INPUT="${CANDIDATE_IK_JSON}"
  else
    echo "Error: input is raw_trajectories, but matching IK JSON was not found: ${CANDIDATE_IK_JSON}"
    exit 1
  fi
fi

ACTION_NAME="$(basename "$(dirname "$(dirname "${INPUT}")")" | tr '_-' '  ')"
if [[ -z "${PROMPT:-}" ]]; then
  export PROMPT="A person performs a ${ACTION_NAME} motion with natural full-body movement."
fi

mkdir -p "${OUT_DIR}"

echo "[0/6] Extract HUMI IK ${IK_POSE_SOURCE} poses ..."
PYTHONPATH=. python scripts/ik_recomputed_json_to_pose_json.py \
  --input "${INPUT}" \
  --output "${EXTRACTED_JSON}" \
  --pose-source "${IK_POSE_SOURCE}"

export OUT_DIR
export SHOW_IK_GT="${SHOW_IK_GT:-0}"
if [[ "${SHOW_IK_GT}" == "1" ]]; then
  export IK_JSON_PATH="${INPUT}"
fi

echo "Using flowmatch-clean distill config: ${DISTILL_CONFIG}"
echo "Using flowmatch-clean distill ckpt:   ${DISTILL_CKPT}"

bash scripts/run_humi_to_g1_full_pipeline_flowmatch_clean.sh "${EXTRACTED_JSON}" "${RUN_NAME}"
