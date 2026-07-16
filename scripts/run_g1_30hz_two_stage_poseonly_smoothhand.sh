#!/usr/bin/env bash
set -euo pipefail

cd /mnt/pfs/scalelab/chengqishi/kimodo_my

export PYTHONPATH=/mnt/pfs/scalelab/chengqishi/kimodo_my
export HF_HOME=/mnt/pfs/scalelab/yzh/kimodo_my/huggingface
export HUGGINGFACE_HUB_CACHE=/mnt/pfs/scalelab/yzh/kimodo_my/huggingface/hub
export TRANSFORMERS_CACHE=/mnt/pfs/scalelab/yzh/kimodo_my/huggingface/transformers
export TORCH_HOME=/mnt/pfs/scalelab/chengqishi/.cache/torch
export OMP_NUM_THREADS=4
export MUJOCO_GL=egl

STAGE1_CONFIG=kimodo/distillation/configs/distill_g1_30hz_poseonly_stage1_10k.yaml
STAGE2_CONFIG=kimodo/distillation/configs/distill_g1_30hz_dagger_smoothhand_stage2_20k.yaml

mkdir -p logs outputs/g1_30hz_poseonly_stage1_10k_wristdense_root_sparse_online outputs/g1_30hz_dagger_smoothhand_stage2_20k_wristdense_root_sparse_online

echo "[stage1] pose-only dense wrist/root-sparse distillation"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}" \
/mnt/pfs/scalelab/chengqishi/conda_envs/kimodo/bin/python3.10 -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=2 \
  scripts/train_distill_g1_100_to_20.py \
  --config "${STAGE1_CONFIG}" \
  2>&1 | tee -a logs/g1_30hz_two_stage_poseonly_smoothhand_stage1_online.log

echo "[stage2] DAgger smooth-hand fine-tune"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}" \
/mnt/pfs/scalelab/chengqishi/conda_envs/kimodo/bin/python3.10 -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=2 \
  scripts/train_distill_g1_100_to_20_dagger_teacher_gt_selfacc.py \
  --config "${STAGE2_CONFIG}" \
  2>&1 | tee -a logs/g1_30hz_two_stage_poseonly_smoothhand_stage2_online.log
