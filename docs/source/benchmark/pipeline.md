# Evaluation Pipeline

This page describes the full benchmark workflow, which uses scripts in the `benchmark` directory:

1. Build full test suite using ground-truth motions from BONES-SEED BVH data and benchmark metadata (`create_benchmark.py`),
2. Generate motions with a model for all or part of the test suite (`generate_eval.py`),
3. Compute text/motion embeddings with pre-trained TMR model (`embed_folder.py `),
4. Evaluate metrics over all generated samples (`evaluate_folder.py`),
5. Aggregate and summarize results (`parse_folder.py`).

This pipeline works off-the-shelf for Kimodo models. To evaluate your own model, step (2) will need to be modified to generate with your custom model and output in the expected npz format.

## Prerequisite: Download Motion Data and Metadata
The benchmark is constructed from motions in the BONES-SEED dataset and our released metadata. Make sure you have downloaded the [BONES-SEED dataset](https://huggingface.co/datasets/bones-studio/seed) along with the metadata for the test suite from HuggingFace at [`nvidia/Kimodo-Motion-Gen-Benchmark`](https://huggingface.co/datasets/nvidia/Kimodo-Motion-Gen-Benchmark). 

The `testsuite` folder from the downloaded metadata contains the directory structure described in the [benchmark introduction](introduction.md) with `meta.json`, `seed_motion.json`, and `seed_constraints.json` metadata files in the leaf folders. These metadata files contain info about the text prompts, durations, and constraint definitions for each test case. The first two steps of the evaluation pipeline will create the following in the leaf folders to prepare for computing metrics:

- **Ground-Truth Motion** (`gt_motion.npz`): produced by `create_benchmark.py` from SEED BVH + metadata.
- **Constraints Configuration** (`constraints.json`): for test cases with constraint inputs, this file is created by `create_benchmark.py` from SEED BVH + metadata.
- **Generated Motion** (`motion.npz`): produced by the generation step from the model to evaluate (e.g. `generate_eval.py`).

To perform the full evaluation, including metrics for both ground-truth and generated motions (steps 3--5), each leaf folder must contain both `gt_motion.npz` and `motion.npz`.

> Note: all of the following steps will work with a _subset_ of the full test suite, if desired. Anywhere the `testsuite` directory is passed in, it can be replaced with a specific subset such as `testsuite/content/text2motion` to only run this subset of the benchmark.

## 1. Build Full Benchmark (`create_benchmark.py`)

 The `create_benchmark.py` script bridges the ground truth motions and metadata: it downloads the testsuite structure (if not already present locally), then reads the referenced BVH files from a local copy of BONES-SEED and writes `gt_motion.npz` and `constraints.json` into each sample folder.

```bash
python benchmark/create_benchmark.py path/to/testsuite --dataset datasets/bones-seed/soma_uniform
```

By default, this construction can take several hours and the resulting folder is about **26 GB**. 

To run faster, you can increase the number of parallel workers for processing:
```bash
OMP_NUM_THREADS=2 python benchmark/create_benchmark.py path/to/testsuite --dataset datasets/bones-seed/soma_uniform --workers 16
```
This example runs well with a 32-core system, but you may need to adjust the number of threads-per-worker and total workers for your system. Generally, a lower number of threads-per-worker with larger number of workers (up to your available CPU capacity) runs fastest.

Options:

- `--dataset`: path to the local SEED dataset folder (default: `datasets/bones-seed/soma_uniform`).
- `--workers`: number of parallel workers to use for benchmark construction (default: 1, sequential)
- `--overwrite`: rebuild `gt_motion.npz` even if it already exists.

For each test case, the script:

1. parses the BVH file into local rotation matrices and root translation,
2. subsamples to 30 FPS,
3. converts to the standard T-pose via `SOMASkeleton77.to_standard_tpose`,
4. computes Kimodo motion features and canonicalizes the motion,
5. writes the resulting motion dictionary as `gt_motion.npz`.

For a detailed walkthrough of steps 1--4, see [Loading BONES-SEED BVH data](../user_guide/seed_dataset.md).

## 2. Generate Motions (`generate_eval.py`)

The next step is to generate a motion for each test case.
The script `benchmark/generate_eval.py` recursively generates one motion with Kimodo per test case from either the full `testsuite` or a  desired subset. 

```bash
python benchmark/generate_eval.py \
  --benchmark path/to/testsuite \
  --output generated_folder \
  --model kimodo-soma-rp \
  --batch_size 32 \
  --num_workers 4
```

The batch size and number of data workers should be adjusted for your system. The script is intended to be run with the latest Kimodo-SOMA models (right now v1.1) which are compatible with the benchmark.

> Note: each test cases has a seed in `meta.json` that is  loaded and used for generation to enable reproducibility. However, by default, the generation script uses the first seed in a batch to seed the whole batch, so to make results completely repeatable, you must set the batch size to 1 or always use the same batch size when running generation.

Useful options:

- `--model`: Kimodo model to use for generation. See [available models](../getting_started/quick_start.md#overview-kimodo-models) for the full list. 
- `--output`: output root directory. The testsuite hierarchy is mirrored here. If omitted, motions are generated **in-place** inside the testsuite folder.
- `--overwrite`: regenerate even if `motion.npz` already exists.
- `--diffusion_steps`: default denoising steps (can be overridden by each sample `meta.json`).
- `--postprocess`: enable post-processing. For fair evaluation, it is recommended to **not** use post-processing so that metrics reflect the raw model output.
- `--text_encoder_fp32`: will instantiate the text encoder (if needed) with float32 precision instead of bfloat16. The Kimodo v1.1 models are trained with float32 text encodings, so this slightly improves accuracy but requires extra VRAM.

After generation, the output tree mirrors the `testsuite` hierarchy and includes generated motions (`motion.npz`). If the testsuite was built with `create_benchmark.py`, each leaf already has `gt_motion.npz`; the generation step adds `motion.npz` per sample.

```text
generated_folder/
└── .../0000/
    ├── meta.json
    ├── constraints.json                # present if available in testsuite
    ├── gt_motion.npz                   # if built with create_benchmark
    └── motion.npz                      # generated
```

### Using Custom Models

The `generate_eval` script is set up to work with Kimodo models, but it can be easily adapted or replaced by generation with a custom model. The only requirement to be able to compute all metrics is to output the `motion.npz` file for each test case that minimally contains: (1) `posed_joints` field with global joint positions on the SOMA 77-joint skeleton and (2) `foot_contacts` field with binary foot contact predictions. Please see the [output formats docs](../user_guide/output_formats.md) for more details on the `NPZ` format.

## 3. Embed with Pre-Trained TMR (`embed_folder.py`)

Several evaluation metrics such as R-precision, FID, and latent similarity rely on latent embeddings of both motion and text. For this purpose, we use a [Text-Motion-Retrieval (TMR)](https://mathis.petrovich.fr/tmr/) model trained on the full Bones Rigplay dataset. See [Metrics](metrics.md) for details on the TMR evaluation protocol and metrics. 

The next step in the eval pipeline is using this TMR model with the `benchmark/embed_folder.py` script to recursively embed each generated motion (`motion.npz`), GT motion (`gt_motion.npz`) when present, and the text prompt from `meta.json`:

```bash
python benchmark/embed_folder.py generated_folder --model tmr-soma-rp
```

The default TMR model (`tmr-soma-rp`) trained on the full Rigplay dataset is released as [`TMR-SOMA-RP-v1`](https://huggingface.co/nvidia/TMR-SOMA-RP-v1). It is automatically downloaded from HuggingFace on first use of the embedding script. 

Options:

- `--model`: TMR model to use for encoding (default: `tmr-soma-rp`).
- `--device`: compute device (`auto`, `mps`, `cuda`, or `cpu`). `auto` prefers CUDA, then MPS, then CPU.
- `--overwrite`: re-embed even if embedding files already exist.
- `--text_encoder_fp32`: will instantiate the text encoder (if needed) with float32 precision instead of bfloat16. The TMR model is trained with float32 text encodings, so this slightly improves accuracy but requires extra VRAM.

Running this script saves the embeddings to each test case folder that has the corresponding motion file(s) and `meta.json`:

- `motion_embedding.npy` (when `motion.npz` exists)
- `gt_motion_embedding.npy` (when `gt_motion.npz` exists)
- `text_embedding.npy`

> Note: this script can take over 1 hour to run for the full test suite, depending on your GPU.

## 4. Compute Evaluation Metrics (`evaluate_folder.py`)

Next, use `benchmark/evaluate_folder.py` to compute per-test-case and aggregated metrics across the test suite (or a specific subset folder). Each leaf folder must contain both `motion.npz` and `gt_motion.npz` to compute the metrics.

```bash
python benchmark/evaluate_folder.py generated_folder
```

Options:

- `--device`: compute device (`auto`, `mps`, `cuda`, or `cpu`). `auto` prefers CUDA, then MPS, then CPU.

The script runs two evaluation passes: one on the generated motion (`motion.npz`) and one on the ground-truth motion (`gt_motion.npz`). It outputs:

- per test case results: `metrics.json` inside each test case (leaf) folder with metrics summarized for that single test case
- per group results: `<group_name>.json` one level above each group of test-case folders that aggregates metrics over all contained test cases

Please see the [Metrics](metrics.md) page for a detailed explanation of these json formats.

After embedding and evaluation, the folder structure should look like:

```text
generated_folder/
├── .../0000/
│   ├── motion.npz
│   ├── gt_motion.npz
│   ├── motion_embedding.npy
│   ├── gt_motion_embedding.npy
│   ├── text_embedding.npy
│   └── metrics.json              # single test-case metrics
└── .../<group_name>.json         # folder-level aggregate summary of all contained test cases
```

## 5. Summarize Results of Full Benchmark (`parse_folder.py`)

If you have computed metrics for the _entire_ test suite (both `content` and `repetition` splits), use `benchmark/parse_folder.py` to validate all per-test-case result JSONs and aggregate metrics into summary tables. Unlike the previous steps, this script expects the user to pass in the root `testsuite` and for the test suite to follow the standard split/category hierarchy (see [Introduction](introduction.md)):

- **Splits**: `content`, `repetition`
- **Categories**: `overview`, `timeline_single`, `timeline_multi` (text-following), `constraints_withtext`, `constraints_notext` (constrained generation)

```bash
python benchmark/parse_folder.py generated_folder
```

Options:

- `--output`: path for the output JSON (default: `<folder>/summary_rows.json`).
- `--format`: table output format. `terminal` (default) for fixed-width tables, `md` for markdown tables suitable for copy-pasting into documentation.

The script:

1. discovers all grouped test case directories (folders containing single test cases with `meta.json`, `motion.npz`, and `gt_motion.npz`),
2. loads each group's `<group_name>.json` result files written by `evaluate_folder`,
3. computes weighted averages of all metrics by split and category,
4. writes `summary_rows.json` with per-row and per-table aggregated results,
5. prints formatted benchmark tables to the terminal (text-following and constraints, with GT and method rows side by side).

Metric values in the tables are converted to user-friendly units (e.g., constraint position errors in cm, foot skating in cm/s). See [Metrics](metrics.md) for definitions of individual metrics.
