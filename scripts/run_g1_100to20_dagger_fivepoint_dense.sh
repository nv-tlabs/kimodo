#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/mnt/pfs/scalelab/chengqishi/kimodo_my
cd "${ROOT_DIR}"

PYTHON_BIN=${PYTHON_BIN:-/mnt/pfs/scalelab/chengqishi/conda_envs/kimodo/bin/python3.10}
CONFIG_PATH=${CONFIG_PATH:-kimodo/distillation/configs/distill_g1_100_to_20_dagger_teacher_gt03_100k_bs4x4_k3_cosine_selfacc50_rootacc10_headingacc10.yaml}
GPU_LIST=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
RESUME_PATH=${RESUME_PATH:-}
LOG_DIR=${LOG_DIR:-logs}

IFS=',' read -r -a GPU_IDS <<< "${GPU_LIST}"
NPROC_PER_NODE=${NPROC_PER_NODE:-${#GPU_IDS[@]}}
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE=${LOG_FILE:-${LOG_DIR}/g1_100to20_dagger_fivepoint_dense_${RUN_TIMESTAMP}.log}

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME=${HF_HOME:-/mnt/pfs/scalelab/yzh/kimodo_my/huggingface}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}
export TORCH_HOME=${TORCH_HOME:-/mnt/pfs/scalelab/chengqishi/.cache/torch}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}

mkdir -p "${LOG_DIR}"

CMD=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --standalone
  --nnodes=1
  --nproc_per_node="${NPROC_PER_NODE}"
  scripts/train_distill_g1_100_to_20_dagger_teacher_gt_selfacc.py
  --config "${CONFIG_PATH}"
)

if [[ -n "${RESUME_PATH}" ]]; then
  CMD+=(--resume "${RESUME_PATH}")
fi

echo "[distill] GPUs: ${GPU_LIST}"
echo "[distill] workers: ${NPROC_PER_NODE}"
echo "[distill] config: ${CONFIG_PATH}"
echo "[distill] log: ${LOG_FILE}"
if [[ -n "${RESUME_PATH}" ]]; then
  echo "[distill] resume: ${RESUME_PATH}"
fi

CUDA_VISIBLE_DEVICES="${GPU_LIST}" "${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
