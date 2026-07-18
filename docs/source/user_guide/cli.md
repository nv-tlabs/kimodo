# Command-Line Interface

The primary CLI entrypoint is the `kimodo_gen` command. This corresponds to the script located in `kimodo/scripts/generate.py`, therefore you can equivalently use `python -m kimodo.scripts.generate`.

**Docker Usage**: If you set up Kimodo with Docker, you can instead run generation inside the Docker container, replacing `kimodo_gen XXX` with `docker compose run --rm demo kimodo_gen XXX`. If you will be running generation multiple times, it is better to start the `demo` container (e.g., in another terminal or in the background), and then run commands inside it with `docker compose exec demo kimodo_gen XXX`.

**Single Prompt Generation:**

```bash
kimodo_gen "A person walks forward." \
    --model Kimodo-SOMA-RP-v1 \
    --duration 5.0 \
    --output output
```

The `--model` command corresponds to the Kimodo model checkpoint to generate with. By default, the `Kimodo-SOMA-RP-v1` is used if not provided. The output motion will be saved using the stem name given by `--output` in the Kimodo [output format](../user_guide/output_formats.md). If generating with a G1 or SMPL-X model, you can also save to other output formats like MuJoCo qpos CSV file and AMASS NPZ format.

For **offline conversion** between Kimodo NPZ, AMASS NPZ, SOMA BVH, and G1 MuJoCo CSV after generating, use `kimodo_convert` (see [Motion format conversion](motion_convert.md)).

**Multi-Prompt Generation:**

Generating from a sequence of text prompts can be achieved by using multiple sentences separated by periods with corresponding durations:

```bash
kimodo_gen "A person walks forward. A person is walking backwards." \
    --duration "5.0 4.0" \
```

This command will use Kimodo to generate each prompt in sequence, with constraints added to the beginning of the second prompt to ensure continuity with the first generated motion. You can control how many frames are used to blend consecutive motions with the `--num_transition_frames` argument.

**Single Prompt with Constraints:**

Generation can be constrained by providing a constraints JSON file (see the [Constraints Format Definition](constraints.md)).

```bash
kimodo_gen "A person walks forward and picks something up from the ground." \
    --model Kimodo-SOMA-RP-v1 \
    --duration 5.0 \
    --constraints kimodo/assets/demo/examples/kimodo-soma-rp/03_full_body_keyframes/constraints.json
```

Constraint files can be created and saved from the interactive demo or manually defined following
the [constraints format guide](constraints.md).

## Output Formats

For full details on output formats, see [this page](output_formats.md).

To convert between these formats offline, see [Motion format conversion](motion_convert.md) (`kimodo_convert`).

CLI generation uses a single **output stem** (`--output`) for all formats (NPZ, AMASS NPZ, CSV, and BVH). It can write either **one file** or **a folder of files**, depending on the number of samples:

- **One sample** (`--num_samples 1`): writes a single file per format at the stem (e.g. `--output test` → `test.npz`, `test.csv`). No folder is created. For SMPLX, AMASS is written to `test_amass.npz`.
- **Multiple samples**: creates a folder with that stem and writes one file per sample with suffixes `_00`, `_01`, etc. (e.g. `--output test` → `test/test_00.npz`, ...).

Use the `--bvh` flag to also export BVH (SOMA only) to the same stem.

### Output Rest Pose

For SOMA-based Kimodo models, motions can be exported with respect to two different rest poses. The default rest pose, that is always used by the `NPZ` format, is a standard T-pose consistent with the canonical T-pose of the SOMA model. For `BVH` outputs, the default rest pose is a non-standard pose, but it is consistent with the BVH format of the [BONES-SEED dataset](https://huggingface.co/datasets/bones-studio/seed). To output a `BVH` file with the standard T-pose as the rest pose, you can use the `--bvh_standard_tpose` option.

The standard T-pose used by Kimodo is available as a BVH file in the [repo assets](https://github.com/nv-tlabs/kimodo/tree/main/kimodo/assets/skeletons/somaskel77).

## Visualizing Generated Motions

Motions generated with the CLI can be visualized in the demo UI. To do this, under "Load/Save" > "Motion", type in the path of the generated output npz file, then click "Load Motion" to load it into the viewer. If you used constraints when generating, those can also be loaded in in a similar way.

## Arguments

To see all available flags, run `kimodo_gen --help`. They are:

- `prompt`: Text description of the desired motion. Required unless `--input_folder` supplies prompts through `meta.json`.
- `--model`: Model name to use (default alias: `kimodo-soma-rp`, currently resolving to `Kimodo-SOMA-RP-v1.1`; options are the models in [this table](../getting_started/quick_start.md#overview-kimodo-models))
- `--duration`: Motion duration in seconds (default: `5.0`). For multiple prompts,
  pass space-separated durations in a string.
- `--diffusion_steps`: Number of denoising steps (default: `100`)
- `--num_samples`: Number of motion variations to generate (default: `1`)
- `--num_transition_frames`: Frames used to blend between prompts (default: `5`)
- `--constraints`: Path to a JSON file containing constraints
- `--output`: Output stem name (default: `output`). Used for all formats (NPZ, AMASS NPZ, CSV, BVH). With one sample, writes a single file per format (e.g. `test.npz`, `test.csv`). With multiple samples, creates a folder and writes `test_00.npz`, `test_01.npz`, … inside it. For SMPLX with one sample, AMASS is written to `stem_amass.npz` so it does not overwrite the main NPZ.
- `--save_example_dir`: If given, saves outputs to an "example" directory structure that can be loaded in the Kimodo demo.
- `--bvh`: Optional flag. When set, also export BVH (SOMA models only) using the same stem as `--output`.
- `--bvh_standard_tpose`: If exporting BVH, export with the rest pose being the standard T-pose rather than the rest pose consistent with the BONES-SEED dataset.
- `--seed`: Seed for reproducible results
- `--device`: PyTorch device (`auto`, `mps`, `cuda`, or `cpu`; default: `auto`)
- `--no-postprocess`: Disable post-processing (includes foot skate cleanup and constraint optimization)
- `--input_folder`: Folder containing meta.json and optional constraints.json. If set, generation settings are loaded from meta.json. These are found in demo example folders.
- `--cfg_type`: Classifier-free guidance mode: `nocfg`, `regular`, or `separated` (the custom mode with independent text and constraint scales). See {ref}`Classifier-free guidance (details) <classifier-free-guidance-cfg>` below.
- `--cfg_weight`: One float for `regular` CFG, or two floats `[text_weight, constraint_weight]` for `separated` CFG. If you pass only weights (no `--cfg_type`), one value implies `regular` and two imply `separated`. Not used with `nocfg`.

:::{dropdown} Classifier-free guidance (CFG)
:name: classifier-free-guidance-cfg

The CLI mirrors the Python API in [Generation parameters](configuration.md): Kimodo supports standard CFG (`regular`) and a **separated** variant with two scales—text vs. constraints—which is the usual setting in this project.

**Rules:**

- `nocfg`: no weights; do not pass `--cfg_weight`.
- `regular`: pass exactly one value after `--cfg_weight`.
- `separated`: pass exactly two values after `--cfg_weight`.

If you pass **`--cfg_type` or `--cfg_weight` on the command line**, those values override any `cfg` block in `meta.json` when using `--input_folder`. If you omit both flags, `meta.json` may still supply CFG via `cfg.enabled`, `cfg.text_weight`, and `cfg.constraint_weight` (same shape as the interactive demo examples). If there is no CLI CFG and no `cfg` in meta, the model uses its built-in defaults.

Examples:

```bash
# No classifier-free guidance
kimodo_gen "A person walks." --cfg_type nocfg

# Standard CFG (single scale)
kimodo_gen "A person walks." --cfg_type regular --cfg_weight 2.5

# Separated CFG (text scale, then constraint scale)
kimodo_gen "A person walks." --cfg_type separated --cfg_weight 2.0 1.5

# Infer mode from arity: one float -> regular; two floats -> separated
kimodo_gen "A person walks." --cfg_weight 2.0 2.0
```

:::

## Python API
The `kimodo/scripts/generate.py` script is a good place to start to familiarize yourself with the Python API of Kimodo if you'd like to use this directly. The full model API is detailed in the [API documentation](../api_reference/index.rst).

If you want to use kimodo in another project, you can interact with it like this:

```python
from kimodo import load_model

model = load_model("kimodo-soma-rp", device="auto")
output = model(
    prompts="A person jumps",
    num_frames=150,
    num_denoising_steps=100,
)
```
