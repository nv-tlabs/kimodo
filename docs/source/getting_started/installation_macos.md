# Installation on Apple Silicon

Kimodo supports Apple Silicon Macs through PyTorch's Metal Performance Shaders
(MPS) backend. The native `MotionCorrection` extension is compiled for ARM64
and uses SIMDe to translate its existing x86 SIMD code to ARM NEON.

These steps target M1 through M4 systems. The native build and Metal smoke test
were verified on an M4 Max.

## Prerequisites

- macOS 14.0 or newer (required by the default bfloat16 text encoder on MPS)
- Xcode Command Line Tools (`xcode-select --install`)
- Homebrew
- `uv` or another Python environment manager
- Python 3.10 through 3.12
- A Hugging Face account with access to
  [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

Install CMake, create an isolated environment, and install the native macOS
PyTorch wheel before Kimodo:

```bash
git clone https://github.com/nv-tlabs/kimodo.git
cd kimodo

brew install cmake

uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch
uv pip install -e .
```

Do not use `docker_requirements.txt` on macOS; it is locked for Python 3.10 on
x86_64 Linux. Install from `pyproject.toml` with the native environment above.
The repository's NVIDIA Docker image cannot expose Metal through Docker
Desktop, so M4 GPU acceleration requires this native installation.

SIMDe, pybind11, and Eigen are fetched automatically during the extension
build when suitable system packages are not present. To install the interactive
demo dependencies, replace the final command with:

```bash
uv pip install -e ".[all]"
```

Authenticate once so the gated text encoder can be downloaded:

```bash
hf auth login
```

## Verify Metal acceleration

```bash
python -c "import platform; assert platform.machine() == 'arm64'; print(platform.machine())"
python -c "import torch; print(torch.backends.mps.is_available())"
python -c "import motion_correction; print('MotionCorrection: OK')"
python -c "from kimodo.device import resolve_device; print(resolve_device())"
EXTENSION=$(python -c "import motion_correction._motion_correction as m; print(m.__file__)")
file "$EXTENSION"
```

These commands should report `arm64`, `True`, a successful MotionCorrection
import, `mps`, and a `Mach-O 64-bit bundle arm64`, respectively.

The ARM64 MotionCorrection build was compared with the original x86_64 AVX
implementation under Rosetta using a fixed 48-frame fixture. Root translations
matched exactly; quaternion output differed by at most `1.45e-5` and passed a
`2e-5` absolute/relative tolerance.

## Generate on the M4 GPU

```bash
kimodo_gen "A person walks forward." \
  --device mps \
  --duration 3 \
  --diffusion_steps 20 \
  --output output/m4_walk
```

The released checkpoint's denoiser and native post-processing path completed
150 frames and 100 steps on MPS without PyTorch's CPU fallback. That smoke
test used a stub text embedding because the Llama text encoder is gated; the
shown CLI command additionally requires the Hugging Face access configured
above. If a future PyTorch release introduces an unsupported MPS operation,
retry with `PYTORCH_ENABLE_MPS_FALLBACK=1` while diagnosing that operation.

The CLI, demo, benchmark utilities, low-level `load_model()` API, and local
LLM2Vec text encoder all select MPS automatically. Use `--device cpu` or set
`KIMODO_DEVICE=cpu` to override this. The local text encoder follows that
device unless it is placed separately with `TEXT_ENCODER_DEVICE=cpu` or
`TEXT_ENCODER_DEVICE=mps`. The M4 default is
bfloat16; set `TEXT_ENCODER_DTYPE=float16` if an older MPS/PyTorch combination
cannot execute a bfloat16 operator.

If the optional MotionCorrection extension cannot be built, a generation-only
fallback is available:

```bash
SKIP_MOTION_CORRECTION_IN_SETUP=1 uv pip install -e .
kimodo_gen "A person walks." --device mps --no-postprocess
```

G1 generation already disables that post-processing path. SOMA and SMPL-X
must use `--no-postprocess` with this fallback.

## M4 Max memory notes

Kimodo and its 8B text encoder require roughly 17 GB of accelerator memory in
the upstream configuration. Apple Silicon uses unified memory, and the current
loader first creates text-encoder weights on CPU before moving them to MPS, so
peak memory during startup can be noticeably higher. Close other memory-heavy
applications and leave substantial headroom for macOS; 32/36 GB configurations
may be tight. Model and text-encoder weights are also downloaded on first use.

If local text encoding is too memory-intensive, run the encoder on a second
machine on a private network (or through an SSH tunnel):

```bash
GRADIO_SERVER_NAME=0.0.0.0 kimodo_textencoder
```

Then set `TEXT_ENCODER_MODE=api` and
`TEXT_ENCODER_URL=http://private-host:9550/` on the Mac. This service has no
authentication; never expose port 9550 directly to the public internet.

Seeds improve repeatability, but MPS and CUDA use different kernels and neither
cross-backend equality nor bitwise determinism is guaranteed.
