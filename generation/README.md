# Generation

Prepare UAV masks, edit the background with FLUX.2-dev, and integrate the source UAV to produce the final image. Supported conditions: rain, snow, fog, dust, and low light.

## Setup

Follow [ENVIRONMENT.md](ENVIRONMENT.md) for dependencies and model downloads. Run the commands below from `generation/`.

## Generate

```bash
python generate.py generate --manifest examples/sources.jsonl \
  --model models/FLUX.2-dev --output outputs/example
```

Use `--conditions fog` to select a condition, or `--dry-run` to check the inputs. Choose a new output directory for each run. Settings and prompts are in [configs/generation.json](configs/generation.json) and [prompts/background.json](prompts/background.json).

To prepare masks for your own data:

```bash
python generate.py prepare-masks --manifest data/sources.jsonl \
  --sam-checkpoint models/sam_vit_b_01ec64.pth --output outputs/masks
python generate.py generate --manifest outputs/masks/sources.jsonl \
  --model models/FLUX.2-dev --output outputs/generated
```

The input format and processing settings are described in [DATA.md](DATA.md).

## Outputs and examples

Final images are saved to `<output>/final/`, annotations to `<output>/annotations/`, and file records to `<output>/manifest.jsonl`.

[examples/README.md](examples/README.md) lists the complete fog workflow and the individual rain, snow, fog, and dust results.

## Checks

```bash
python tools/validate.py
sha256sum -c checksums.sha256
```
