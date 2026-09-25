# Environment and models

Use Python 3.12 and a CUDA GPU with BF16 support. Package versions are recorded in [environment.json](environment.json); source revisions are pinned in [requirements.txt](requirements.txt).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

The CPU checks use Pillow, NumPy, and OpenCV. Generation uses FLUX.2-dev with model CPU offload.

## FLUX.2-dev

Download `black-forest-labs/FLUX.2-dev` at revision `26afe3a78bb242c0a8bb181dcc8937bb16e5c66c`, using an account with model access:

```bash
hf download black-forest-labs/FLUX.2-dev \
  --revision 26afe3a78bb242c0a8bb181dcc8937bb16e5c66c \
  --include 'model_index.json' 'scheduler/*' 'text_encoder/*' 'tokenizer/*' 'transformer/*' 'vae/*' \
  --local-dir models/FLUX.2-dev
```

Pass the directory with `--model`. The pipeline loads its scheduler and components from this directory.

## SAM

Download the `vit_b` checkpoint `sam_vit_b_01ec64.pth` from [segment-anything](https://github.com/facebookresearch/segment-anything) and pass it with `--sam-checkpoint` when preparing masks. Save it under `models/` or another local directory.

Checkpoint SHA-256:

```text
ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912
```

Inputs with protection masks can be passed directly to `generate`.

## Third-party notices

Dependency license notices are in [licenses](licenses). Model licenses accompany their respective distributions.
