# Installation

> Note: This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

> Note: This repo was developed and primarily tested on Linux

> Apple Silicon users should follow the native [macOS installation guide](installation_macos.md). Docker Desktop cannot expose Apple's MPS backend.

There are two ways to install Kimodo: (1) as a package, or (2) download the source code and install.
Both require setting up a Hugging Face token to use the text encoder at generation time.

## Set Up Hugging Face Token

The Kimodo text encoder relies on the **gated** `meta-llama/Meta-Llama-3-8B-Instruct` model, which requires:
- Your HF account has been granted access to the [model page](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).
- You provide a HF token for runtime

After receiving access to the Llama repo, please create an access token [here](https://huggingface.co/settings/tokens/new?tokenType=read). Then use it to log in on your command line:

```bash
hf auth login
```
or alternatively, paste the token in this file ``~/.cache/huggingface/token``. If you don't have `hf` installed, you will first need to run `pip install --upgrade huggingface_hub`.

## Kimodo Install Option 1: Package Install

The easiest way to get started is simply installing Kimodo as a package without needing to clone the codebase. This will allow you to generate motions and run the demo as a black box.

We suggest creating a new Python environment for the install, for example with `venv` or conda:
```bash
conda create -n kimodo python=3.10
conda activate kimodo
```

To ensure you have a version of [PyTorch](https://pytorch.org/get-started/locally/) that is compatible with your system and CUDA version, it is recommended to manually install the best version of PyTorch for you before installing Kimodo. Anything over PyTorch 2.0 is sufficient. We strongly suggest using a GPU-capable version of PyTorch to generate motions in a reasonable amount of time.

Installing the base Kimodo package will allow you generate motions with the command line:
```bash
pip install git+https://github.com/nv-tlabs/kimodo.git
```

If you want to be able to run the interactive demo as well, use this command which installs additional dependencies:
```bash
pip install "kimodo[all] @ git+https://github.com/nv-tlabs/kimodo.git"
```

Now should be ready to use Kimodo. Check out the [quick start guide](quick_start.md) to see how to generate motions.

If you experience package or system compatibility issues, use the source-code route below. Linux users can then choose Docker; Apple Silicon users should use the native macOS guide for MPS acceleration.

## Kimodo Install Option 2: Source Code Install

If you plan to build on Kimodo or dig into the codebase, you'll want to clone and install the repo.

### Clone Kimodo Repository

```bash
git clone https://github.com/nv-tlabs/kimodo.git
cd kimodo
```

### Choose Your Installation Route
Kimodo can be installed by building and running through a virtual environment (e.g., `conda`) or within a Docker container.

```{toctree}
:maxdepth: 1

installation_virtual_env
installation_macos
installation_docker
installation_smpl
```
