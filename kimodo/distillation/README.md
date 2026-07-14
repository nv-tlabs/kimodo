# Kimodo Distillation (16->8, 100->20)

## Goal
- Teacher: existing Kimodo G1 16-layer denoiser (pretrained checkpoint)
- Student: 8-layer denoiser
- Distillation target: student trained on reduced 20-step timestep grid while matching teacher behavior trained on full schedule

## Loss (7+7)
- Teacher branch (main): 7-term Kimodo loss between student `pred_x0` and teacher `pred_x0`
- GT branch (aux): 7-term Kimodo loss between student `pred_x0` and dataset GT `x0`
- Final:
  - `L = teacher_weight * L_teacher7 + gt_weight * L_gt7`
  - default: `teacher_weight=0.8`, `gt_weight=0.2`

## Files
- `kimodo/distillation/loss.py`: 7+7 weighted loss
- `kimodo/distillation/train.py`: distillation training loop
- `kimodo/distillation/configs/distill_g1_100_to_20.yaml`: default config
- `scripts/train_distill_g1_100_to_20.py`: python entrypoint
- `scripts/run_distill_g1_100_to_20.sh`: shell wrapper

## Run
```bash
export HF_HOME=./huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
bash scripts/run_distill_g1_100_to_20.sh
```

Optional resume:
```bash
RESUME_PATH=outputs/g1_distill_16to8_100to20/checkpoints/step_00002000.pt \
  bash scripts/run_distill_g1_100_to_20.sh
```

## Notes
- `distillation.student_steps=20` controls the timestep grid used during distillation.
- `distillation.teacher_steps=100` is metadata for experiment tracking; teacher itself remains full-capacity pretrained.
- Student warm start from teacher is enabled via same-name same-shape parameter copy.

## Two-stage Training

### Stage 1: Offline Distillation

配置：`kimodo/distillation/configs/distill_g1_100_to_20_schedule.yaml`

从固定 motion 数据集构造加噪状态，使用 Teacher 输出和数据集 GT 共同监督 Student。训练
10,000 step，最终生成：

```text
outputs/g1_distill_16to8_100to20_schedule/ema_final.pt
```

两卡启动示例（`batch_size=64` 是单卡 batch，global batch 为 128）：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 \
  -m kimodo.distillation.train \
  --config kimodo/distillation/configs/distill_g1_100_to_20_schedule.yaml
```

### Stage 2: DAgger Distillation

配置：`kimodo/distillation/configs/distill_g1_100_to_20_dagger_teacher_gt03_100k_bs4x4_k3_cosine_selfacc50_rootacc10_headingacc10.yaml`

从 Stage 1 的 `ema_final.pt` 初始化 Student。Student 执行 20-step rollout，每条 rollout
抽取 3 个访问状态，由 Teacher 重新标注，再结合 GT 和 acceleration loss 训练 100,000 step。

四卡启动示例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 \
  scripts/train_distill_g1_100_to_20_dagger_teacher_gt_selfacc.py \
  --config kimodo/distillation/configs/distill_g1_100_to_20_dagger_teacher_gt03_100k_bs4x4_k3_cosine_selfacc50_rootacc10_headingacc10.yaml
```

DAgger 必须使用专用脚本 `train_distill_g1_100_to_20_dagger_teacher_gt_selfacc.py`；基础
`kimodo.distillation.train` 不会执行 Student rollout。
