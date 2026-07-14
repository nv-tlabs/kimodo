#!/usr/bin/env bash
set -euo pipefail

# End-to-end pipeline for one raw HUMI trajectory JSON using the DDIM-final-clean
# Flow-Matching student:
# 1) Convert HUMI JSON -> ee-pose adapter NPZ
# 2) Convert adapter NPZ -> Kimodo ee-pose constraints JSON
# 3) Compute duration
# 4) Resolve/use prompt and generate G1 motion with continuous FM ODE sampling
# 5) Validate constraints against generated MuJoCo qpos CSV
# 6) (Optional) open viser overlay viewer

INPUT="${1:-hf_humi_raw/proposal/raw_trajectories/recording_000.json}"
if [[ -d "${INPUT}" ]]; then
  JSON_PATH="${INPUT%/}/raw_trajectories/recording_000.json"
elif [[ -f "hf_humi_raw/${INPUT}/raw_trajectories/recording_000.json" ]]; then
  JSON_PATH="hf_humi_raw/${INPUT}/raw_trajectories/recording_000.json"
else
  JSON_PATH="${INPUT}"
fi

ACTION_DIR="$(basename "$(dirname "$(dirname "${JSON_PATH}")")")"
if [[ "$(basename "$(dirname "${JSON_PATH}")")" != "raw_trajectories" ]]; then
  ACTION_DIR="$(basename "$(dirname "${JSON_PATH}")")"
fi
RUN_NAME="${2:-${ACTION_DIR}_$(basename "${JSON_PATH}" .json)}"
KEYFRAME_STEP="${KEYFRAME_STEP:-10}"
TARGET_FPS="${TARGET_FPS:-30}"
SOURCE_FPS="${SOURCE_FPS:-}"
TIMELINE_JSONL="${TIMELINE_JSONL:-}"
SKIP_VIEWER="${SKIP_VIEWER:-0}"
OUT_DIR="${OUT_DIR:-scripts/pipeline_outputs/${RUN_NAME}}"
VISER_PORT="${VISER_PORT:-8080}"
ORIGINAL_COLOR="${ORIGINAL_COLOR:-0,200,0}"
GENERATED_COLOR="${GENERATED_COLOR:-255,120,0}"
GHOST_OPACITY="${GHOST_OPACITY:-0.35}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-}"
INFERENCE_STEPS="${INFERENCE_STEPS:-}"
ODE_SOLVER="${ODE_SOLVER:-heun}"
HARD_PROJECT_OBSERVED="${HARD_PROJECT_OBSERVED:-0}"
HARD_PROJECT_PREFIX_FRAMES="${HARD_PROJECT_PREFIX_FRAMES:-0}"
HARD_PROJECT_RELEASE_FRAMES="${HARD_PROJECT_RELEASE_FRAMES:-0}"
ROOT_CONSTRAINT_MODE="${ROOT_CONSTRAINT_MODE:-xyzyaw}"
DISTILL_CONFIG="${DISTILL_CONFIG:-kimodo/distillation/configs/flowmatch_g1_ddim_clean_teacher100_gt_smoke_1gpu.yaml}"
DISTILL_CKPT="${DISTILL_CKPT:-}"
CFG_TYPE="${CFG_TYPE:-separated}"
CFG_WEIGHT="${CFG_WEIGHT:-2.0 2.0}"
MAX_GENERATED_FRAMES="${MAX_GENERATED_FRAMES:-300}"
MAX_DURATION_SEC="${MAX_DURATION_SEC:-}"
HUMI_SEGMENT_MODE="${HUMI_SEGMENT_MODE:-random}"
HUMI_SEGMENT_START_SEC="${HUMI_SEGMENT_START_SEC:-}"
HUMI_SEGMENT_SEED="${HUMI_SEGMENT_SEED:-}"
HUMI_CANONICALIZE="${HUMI_CANONICALIZE:-1}"

if [[ "${ROOT_CONSTRAINT_MODE}" != "xyzyaw" && "${ROOT_CONSTRAINT_MODE}" != "root2d" && "${ROOT_CONSTRAINT_MODE}" != "none" ]]; then
  echo "Error: ROOT_CONSTRAINT_MODE must be one of: xyzyaw, root2d, none"
  exit 1
fi

if [[ "${HUMI_SEGMENT_MODE}" != "random" && "${HUMI_SEGMENT_MODE}" != "start" && "${HUMI_SEGMENT_MODE}" != "fixed" ]]; then
  echo "Error: HUMI_SEGMENT_MODE must be one of: random, start, fixed"
  exit 1
fi
if [[ "${HUMI_SEGMENT_MODE}" == "fixed" && -z "${HUMI_SEGMENT_START_SEC}" ]]; then
  echo "Error: HUMI_SEGMENT_START_SEC must be set when HUMI_SEGMENT_MODE=fixed"
  exit 1
fi
if [[ "${HUMI_CANONICALIZE}" != "1" && "${HUMI_CANONICALIZE}" != "0" ]]; then
  echo "Error: HUMI_CANONICALIZE must be 1 or 0"
  exit 1
fi
if [[ "${CFG_TYPE}" != "separated" && "${CFG_TYPE}" != "regular" && "${CFG_TYPE}" != "nocfg" ]]; then
  echo "Error: CFG_TYPE must be one of: separated, regular, nocfg"
  exit 1
fi
if [[ "${CFG_TYPE}" == "nocfg" ]]; then
  echo "Error: flowmatch-clean continuous sampler does not support CFG_TYPE=nocfg"
  exit 1
fi
if [[ "${ODE_SOLVER}" != "euler" && "${ODE_SOLVER}" != "heun" ]]; then
  echo "Error: ODE_SOLVER must be one of: euler, heun"
  exit 1
fi
if [[ "${HARD_PROJECT_OBSERVED}" != "1" && "${HARD_PROJECT_OBSERVED}" != "0" ]]; then
  echo "Error: HARD_PROJECT_OBSERVED must be 1 or 0"
  exit 1
fi
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

IK_JSON_PATH="${IK_JSON_PATH:-}"
IK_MJCF_PATH="${IK_MJCF_PATH:-}"
SHOW_IK_GT="${SHOW_IK_GT:-0}"
if [[ "${SHOW_IK_GT}" == "1" && -z "${IK_JSON_PATH}" && "$(basename "$(dirname "${JSON_PATH}")")" == "raw_trajectories" ]]; then
  CANDIDATE_IK_JSON="$(dirname "$(dirname "${JSON_PATH}")")/ik_recomputed/$(basename "${JSON_PATH}")"
  if [[ -f "${CANDIDATE_IK_JSON}" ]]; then
    IK_JSON_PATH="${CANDIDATE_IK_JSON}"
  fi
fi

ACTION_NAME="$(basename "$(dirname "$(dirname "${JSON_PATH}")")" | tr '_-' '  ')"
if [[ "$(basename "$(dirname "${JSON_PATH}")")" != "raw_trajectories" ]]; then
  ACTION_NAME="$(basename "$(dirname "${JSON_PATH}")" | tr '_-' '  ')"
fi
DEFAULT_PROMPT="A person performs a ${ACTION_NAME} motion with natural full-body movement."

if [[ -z "${PROMPT:-}" ]]; then
  if [[ -n "${TIMELINE_JSONL}" && -f "${TIMELINE_JSONL}" ]]; then
    echo "Resolving overview prompt from timeline ..."
    PROMPT="$(PYTHONPATH=. python scripts/resolve_timeline_overview_prompt.py \
      --timeline_jsonl "${TIMELINE_JSONL}" \
      --clip "${JSON_PATH}" \
      --fallback "${DEFAULT_PROMPT}" \
      --print_source)"
  else
    echo "Using fallback prompt derived from HUMI action name."
    PROMPT="${DEFAULT_PROMPT}"
  fi
else
  echo "Using prompt from PROMPT environment override."
fi

mkdir -p "${OUT_DIR}"

ADAPTER_NPZ="${OUT_DIR}/source_ee_pose_adapter.npz"
CONSTRAINTS_JSON="${OUT_DIR}/constraints_ee_pose.json"
GEN_CONSTRAINTS_JSON="${OUT_DIR}/constraints_for_generation.json"
OUTPUT_STEM="${OUT_DIR}/g1_generated"
CSV_PATH="${OUTPUT_STEM}.csv"
XML_PATH="kimodo/assets/skeletons/g1skel34/xml/g1.xml"
SEGMENT_JSON="${OUT_DIR}/source_segment.json"
SOURCE_JSON_PATH="${JSON_PATH}"
IK_VIEWER_JSON_PATH="${IK_JSON_PATH}"

if [[ -n "${MAX_GENERATED_FRAMES}" || -n "${MAX_DURATION_SEC}" ]]; then
  echo "Preparing HUMI segment (${HUMI_SEGMENT_MODE}) ..."
  SEGMENT_INFO=$(python - <<PY
import json
import math
import os
import random
from pathlib import Path

src = Path("${JSON_PATH}")
dst = Path("${SEGMENT_JSON}")
with src.open("r", encoding="utf-8") as f:
    data = json.load(f)
episode = data.get("episode")
if not isinstance(episode, list) or not episode:
    raise ValueError(f"Expected non-empty episode in {src}.")

timestamps = [float(x["timestamp"]) for x in episode]
dt = [b - a for a, b in zip(timestamps, timestamps[1:]) if b > a]
if "${SOURCE_FPS}":
    src_fps = float("${SOURCE_FPS}")
elif dt:
    src_fps = float(1.0 / (sum(dt) / len(dt)))
else:
    raise ValueError("Need at least two increasing timestamps to infer source FPS.")

if "${MAX_GENERATED_FRAMES}":
    max_sec = float("${MAX_GENERATED_FRAMES}") / float("${TARGET_FPS}")
else:
    max_sec = float("${MAX_DURATION_SEC}")

total_sec = max(0.0, timestamps[-1] - timestamps[0])
mode = "${HUMI_SEGMENT_MODE}"
if "${HUMI_SEGMENT_START_SEC}":
    start_sec = float("${HUMI_SEGMENT_START_SEC}")
elif mode == "start":
    start_sec = 0.0
elif total_sec <= max_sec:
    start_sec = 0.0
else:
    seed_text = "${HUMI_SEGMENT_SEED}"
    rng = random.Random(int(seed_text)) if seed_text else random.Random()
    start_sec = rng.uniform(0.0, total_sec - max_sec)

start_abs = timestamps[0] + max(0.0, start_sec)
end_abs = start_abs + max_sec
start_idx = next((i for i, t in enumerate(timestamps) if t >= start_abs), len(episode) - 1)
end_idx = start_idx
while end_idx < len(episode) and timestamps[end_idx] < end_abs:
    end_idx += 1
end_idx = max(start_idx + 1, min(end_idx, len(episode)))

out = dict(data)
out["episode"] = episode[start_idx:end_idx]
out.setdefault("metadata", {})
if isinstance(out["metadata"], dict):
    out["metadata"]["source_json_path"] = str(src)
    out["metadata"]["segment_start_index"] = start_idx
    out["metadata"]["segment_end_index_exclusive"] = end_idx
    out["metadata"]["segment_start_sec"] = float(timestamps[start_idx] - timestamps[0])
    out["metadata"]["segment_duration_sec"] = float(timestamps[end_idx - 1] - timestamps[start_idx]) if end_idx > start_idx else 0.0
    out["metadata"]["segment_mode"] = mode

dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", encoding="utf-8") as f:
    json.dump(out, f)

actual_duration = float(timestamps[end_idx - 1] - timestamps[start_idx]) if end_idx > start_idx else 0.0
print(f"{dst}|{start_idx}|{end_idx}|{timestamps[start_idx] - timestamps[0]:.6f}|{actual_duration:.6f}|{src_fps:.6f}")
PY
)
  IFS='|' read -r SOURCE_JSON_PATH HUMI_SEGMENT_START_IDX HUMI_SEGMENT_END_IDX HUMI_SEGMENT_ACTUAL_START_SEC HUMI_SEGMENT_ACTUAL_DURATION_SEC HUMI_SOURCE_FPS <<< "${SEGMENT_INFO}"
  echo "Selected HUMI segment: source_frames=[${HUMI_SEGMENT_START_IDX}, ${HUMI_SEGMENT_END_IDX}), start_sec=${HUMI_SEGMENT_ACTUAL_START_SEC}, duration_sec=${HUMI_SEGMENT_ACTUAL_DURATION_SEC}, source_fps=${HUMI_SOURCE_FPS}"
  if [[ -n "${IK_JSON_PATH}" ]]; then
    IK_VIEWER_JSON_PATH="${OUT_DIR}/source_ik_segment.json"
    python - <<PY
import json
from pathlib import Path

src = Path("${IK_JSON_PATH}")
dst = Path("${IK_VIEWER_JSON_PATH}")
with src.open("r", encoding="utf-8") as f:
    data = json.load(f)
episode = data.get("episode")
if not isinstance(episode, list) or not episode:
    raise ValueError(f"Expected non-empty episode in {src}.")
start = int("${HUMI_SEGMENT_START_IDX}")
end = int("${HUMI_SEGMENT_END_IDX}")
out = dict(data)
out["episode"] = episode[start:min(end, len(episode))]
if "${IK_MJCF_PATH}":
    out["mjcf_path"] = "${IK_MJCF_PATH}"
out.setdefault("metadata", {})
if isinstance(out["metadata"], dict):
    out["metadata"]["source_json_path"] = str(src)
    out["metadata"]["segment_start_index"] = start
    out["metadata"]["segment_end_index_exclusive"] = min(end, len(episode))
dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", encoding="utf-8") as f:
    json.dump(out, f)
print(f"Saved IK viewer segment: {dst}")
PY
  fi
elif [[ -n "${IK_JSON_PATH}" && -n "${IK_MJCF_PATH}" ]]; then
  IK_VIEWER_JSON_PATH="${OUT_DIR}/source_ik_viewer.json"
  python - <<PY
import json
from pathlib import Path

src = Path("${IK_JSON_PATH}")
dst = Path("${IK_VIEWER_JSON_PATH}")
with src.open("r", encoding="utf-8") as f:
    data = json.load(f)
data["mjcf_path"] = "${IK_MJCF_PATH}"
dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", encoding="utf-8") as f:
    json.dump(data, f)
print(f"Saved IK viewer JSON with mjcf override: {dst}")
PY
fi

echo "[1/6] Convert HUMI JSON to ee-pose adapter NPZ ..."
SOURCE_FPS_FLAGS=()
if [[ -n "${SOURCE_FPS}" ]]; then
  SOURCE_FPS_FLAGS+=(--fps "${SOURCE_FPS}")
fi
CANONICALIZE_FLAGS=()
VIEWER_CANONICALIZE_FLAGS=()
if [[ "${HUMI_CANONICALIZE}" == "0" ]]; then
  CANONICALIZE_FLAGS+=(--no-canonicalize)
  VIEWER_CANONICALIZE_FLAGS+=(--no-canonicalize-original)
fi
PYTHONPATH=. python scripts/humi_json_to_ee_pose_npz.py \
  --input "${SOURCE_JSON_PATH}" \
  --output "${ADAPTER_NPZ}" \
  "${SOURCE_FPS_FLAGS[@]}" \
  "${CANONICALIZE_FLAGS[@]}"

echo "[2/6] Convert adapter NPZ to ee-pose constraints JSON ..."
PYTHONPATH=. python scripts/npz_to_ee_pose_constraints.py \
  --input "${ADAPTER_NPZ}" \
  --output "${CONSTRAINTS_JSON}" \
  --target-fps "${TARGET_FPS}" \
  --keyframe-step "${KEYFRAME_STEP}" \
  --root-constraint-mode "${ROOT_CONSTRAINT_MODE}"
cp "${CONSTRAINTS_JSON}" "${GEN_CONSTRAINTS_JSON}"

echo "[3/6] Compute duration from adapter NPZ ..."
DURATION=$(python - <<PY
import numpy as np
d=np.load("${ADAPTER_NPZ}", allow_pickle=True)
if "${MAX_GENERATED_FRAMES}":
    print(float("${MAX_GENERATED_FRAMES}") / float("${TARGET_FPS}"))
else:
    print(float(d["root_global_6d"].shape[0]) / float(d["fps"]))
PY
)
echo "duration_sec=${DURATION}"

if [[ -z "${INFERENCE_STEPS}" ]]; then
  if [[ -n "${DIFFUSION_STEPS}" ]]; then
    INFERENCE_STEPS="${DIFFUSION_STEPS}"
  else
    INFERENCE_STEPS=$(python - <<PY
from omegaconf import OmegaConf
cfg = OmegaConf.load("${DISTILL_CONFIG}")
print(int(cfg.distillation.student_steps))
PY
)
  fi
fi

read -r -a CFG_WEIGHT_ARGS <<< "${CFG_WEIGHT}"
CFG_FLAGS=(--cfg_type "${CFG_TYPE}")
if [[ "${CFG_TYPE}" != "nocfg" ]]; then
  if [[ "${#CFG_WEIGHT_ARGS[@]}" -eq 0 ]]; then
    echo "Error: CFG_WEIGHT must contain at least one value when CFG_TYPE=${CFG_TYPE}"
    exit 1
  fi
  CFG_FLAGS+=(--cfg_weight "${CFG_WEIGHT_ARGS[@]}")
fi
HARD_PROJECT_FLAGS=()
if [[ "${HARD_PROJECT_OBSERVED}" == "1" ]]; then
  HARD_PROJECT_FLAGS+=(--hard_project_observed_motion)
fi
HARD_PROJECT_FLAGS+=(--hard_project_prefix_frames "${HARD_PROJECT_PREFIX_FRAMES}")
HARD_PROJECT_FLAGS+=(--hard_project_release_frames "${HARD_PROJECT_RELEASE_FRAMES}")

echo "[4/6] Generate G1 motion with flowmatch-clean continuous ODE sampler ..."
echo "Prompt: ${PROMPT}"
echo "CFG: type=${CFG_TYPE}, weight=${CFG_WEIGHT}"
echo "DISTILL_CONFIG: ${DISTILL_CONFIG}"
echo "DISTILL_CKPT: ${DISTILL_CKPT}"
echo "INFERENCE_STEPS: ${INFERENCE_STEPS}"
echo "ODE_SOLVER: ${ODE_SOLVER}"
echo "HARD_PROJECT_OBSERVED: ${HARD_PROJECT_OBSERVED}"
PYTHONPATH=. python scripts/generate_g1_with_first_heading_flowmatch_continuous.py "${PROMPT}" \
  --model Kimodo-G1-RP-v1 \
  --duration "${DURATION}" \
  --inference_steps "${INFERENCE_STEPS}" \
  --ode_solver "${ODE_SOLVER}" \
  "${CFG_FLAGS[@]}" \
  "${HARD_PROJECT_FLAGS[@]}" \
  --constraints "${GEN_CONSTRAINTS_JSON}" \
  --heading_source_npz "${ADAPTER_NPZ}" \
  --distill_config "${DISTILL_CONFIG}" \
  --distill_ckpt "${DISTILL_CKPT}" \
  --output "${OUTPUT_STEM}"

if [[ ! -f "${CSV_PATH}" ]]; then
  echo "Error: CSV not found at ${CSV_PATH}"
  exit 1
fi

echo "[5/6] Check constraints on MuJoCo CSV result ..."
PYTHONPATH=. python scripts/check_ee_constraints_mujoco.py \
  --csv "${CSV_PATH}" \
  --constraints "${CONSTRAINTS_JSON}" \
  --xml "${XML_PATH}"

echo "[5.5/6] Check dense HUMI EE tracking on MuJoCo CSV result ..."
PYTHONPATH=. python scripts/check_humi_ee_tracking_mujoco.py \
  --csv "${CSV_PATH}" \
  --humi-json "${SOURCE_JSON_PATH}" \
  --generated-fps "${TARGET_FPS}" \
  --xml "${XML_PATH}" \
  "${CANONICALIZE_FLAGS[@]}"

if [[ "${SKIP_VIEWER}" == "1" ]]; then
  echo "[6/6] Skip viewer (SKIP_VIEWER=1)."
  exit 0
fi

echo "[6/6] Launch viser overlay viewer ..."
IK_VIEWER_FLAGS=()
if [[ -n "${IK_VIEWER_JSON_PATH}" ]]; then
  IK_VIEWER_FLAGS+=(--original-ik-json "${IK_VIEWER_JSON_PATH}")
fi
PYTHONPATH=. python scripts/viser_compare_humi_generated.py \
  --original-json "${SOURCE_JSON_PATH}" \
  "${IK_VIEWER_FLAGS[@]}" \
  "${VIEWER_CANONICALIZE_FLAGS[@]}" \
  --generated-csv "${CSV_PATH}" \
  --generated-npz "${OUTPUT_STEM}.npz" \
  --generated-fps "${TARGET_FPS}" \
  --max-duration-sec "${DURATION}" \
  --port "${VISER_PORT}" \
  --original-color "${ORIGINAL_COLOR}" \
  --generated-color "${GENERATED_COLOR}" \
  --ghost-opacity "${GHOST_OPACITY}"
