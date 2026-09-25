# UAV-CondSelect

Code for **What Should We Generate for UAV Detection?**

UAV-CondSelect selects adverse conditions for generative augmentation of UAV detection. Semantically different conditions can induce similar changes in detector features. We measure paired generated-minus-clean context responses with a frozen clean-trained detector, then use backward greedy selection to balance response coverage against pairwise overlap. A user-specified trade-off weight determines the retained conditions without prescribing the subset size.

## Workflow

1. **Generate paired images.** Edit the background and blend the source UAV through a soft protection mask. The candidate conditions are rain, snow, fog, dust, and low light.
2. **Measure and select.** Compute condition similarities from paired detector responses and select a condition subset using the coverage-overlap objective.
3. **Train and evaluate.** Combine the retained generated images with clean images and train the downstream detector under the specified protocol.

With RT-DETRv2-S, the three-condition selection at lambda = 0.6 improves external mean AP by **2.95 points over Clean-only** and **0.80 points over Full generation**. [Results](downstream/results/README.md) include per-seed AP, condition similarities, selection traces, and additional experiments.

## Quick start

From the repository root, run condition selection on the included RT-DETRv2-S matrix:

```bash
python downstream/tools/select_conditions.py \
  --matrix downstream/results/recomputed/rt_context_selection.json \
  --lambda 0.6 --output outputs/selection.json
```

This command uses Python 3.10 or later and its standard library. It selects **snow, dust, and low light**, prints coverage, overlap, and objective values, and saves the complete deletion trace. Change `--lambda` to explore the trade-off, or use the included YOLO matrix. The [experiment guide](downstream/docs/reproduction.md) connects paper tables to configurations, commands, and result files.

## Repository guide

| Component | Entry point |
| --- | --- |
| Background generation and source-UAV blending | [generation](generation/README.md) |
| Condition selection algorithm | [selection.py](downstream/src/selection.py) |
| Detector training, response extraction, and evaluation | [downstream](downstream/README.md) |
| Experiment configurations and subset lookup | [configuration index](downstream/configs/README.md) |
| Data layout, annotations, and splits | [data guide](downstream/data/README.md) |
| Complete generation workflow and condition examples | [examples](generation/examples/README.md) |
| FLUX.2-klein-base-9B comparison | [Klein](generation/klein/README.md) |

## Environments and resources

Use separate environments for [generation](generation/ENVIRONMENT.md) and [detector experiments](downstream/docs/environment.md). These guides specify dependencies, pretrained model downloads, and hardware. Generation uses one NVIDIA A800 with 80 GB VRAM; downstream training uses one NVIDIA GeForce RTX 5090. The [data guide](downstream/data/README.md) specifies paths for images, masks, frozen references, and response caches.
