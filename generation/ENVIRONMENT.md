# Environment and models

Use Python 3.12. Dependency versions are recorded in [environment.json](environment.json): PyTorch 2.5.1+cu124, torchvision 0.20.1+cu124, diffusers 0.39.0.dev0, transformers 5.9.0, accelerate 1.13.0, Pillow 12.2.0, NumPy 2.4.6, OpenCV 4.13.0.92 and segment-anything 1.0.

Create an isolated environment and install CUDA-compatible PyTorch and torchvision, then the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

[requirements.txt](requirements.txt) pins the diffusers and SAM source commits. CPU validation requires Pillow, NumPy and OpenCV; generation also requires the model dependencies. FLUX generation requires a CUDA GPU with BF16 support, a compatible NVIDIA driver, sufficient GPU memory and host RAM. Model CPU offload is enabled; it does not eliminate GPU or host-memory requirements. Memory demand depends on the model and input grid.

## FLUX.2-dev

Model identifier: `black-forest-labs/FLUX.2-dev`. Revision: `26afe3a78bb242c0a8bb181dcc8937bb16e5c66c`. Obtain access under the model's license and authenticate with your own account. Keep credentials outside the repository.

```bash
hf download black-forest-labs/FLUX.2-dev \
  --revision 26afe3a78bb242c0a8bb181dcc8937bb16e5c66c \
  --include 'model_index.json' 'scheduler/*' 'text_encoder/*' 'tokenizer/*' 'transformer/*' 'vae/*' \
  --local-dir models/FLUX.2-dev
```

The directory must contain `model_index.json`, `scheduler`, `text_encoder`, `tokenizer`, `transformer` and `vae`. Pass it through `--model`. The implementation uses `Flux2Pipeline` with BF16 and model CPU offload. The scheduler is loaded from the model directory (FlowMatchEulerDiscreteScheduler); the code does not override attention or enable quantization.

## SAM

Use the official `vit_b` checkpoint `sam_vit_b_01ec64.pth` from [segment-anything](https://github.com/facebookresearch/segment-anything), saved under `models/` or another configurable location. Checkpoint SHA-256:

```text
ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912
```

Pass the checkpoint using `--sam-checkpoint`. SAM weights are unnecessary when the input manifest already provides protection masks. Model weights are not included in the repository.

## Third-party notices

Dependency notices are included in [licenses](licenses). Model licenses apply separately. The project code and example data require authorization from their respective rights holders before redistribution; no license is assigned on their behalf.
