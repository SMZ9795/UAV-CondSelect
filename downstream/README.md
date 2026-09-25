# UAV-CondSelect downstream experiments

RT-DETRv2-S and YOLO11m training, paired responses, condition selection, COCO evaluation, matched sampling and short-budget transfer. Condition order is rain, snow, fog, dust, low_light; input labels may also use dim_low_light.

Install [numerical dependencies](requirements-numerics.txt). See [environment and weights](docs/environment.md), [protocols](docs/protocols.md), and [extraction and evaluation commands](docs/commands.md).

```bash
python -m pip install -r downstream/requirements-numerics.txt
python downstream/tools/recompute.py --results-only --output results
python downstream/tools/recompute.py --assets artifacts --output recomputed
python downstream/tools/plot_results.py --results recomputed --output figures
```

Arrange images using [image_manifest.csv](data/image_manifest.csv), [response_pairs.csv](data/response_pairs.csv), [transfer_image_manifest.csv](data/transfer_image_manifest.csv), and the file names in [COCO annotations](data/loader_annotations). Keep [split membership](data/splits). Transfer reads train_sources.txt, val_sources.txt and test_sources.txt in DATA_ROOT/rtdetrv2_pilot_dataset/splits/. Copy these three text files from data/splits/. Install response masks from the separate assets archive with tools/install_masks.py.

```bash
python downstream/tools/run.py rt --config downstream/configs/rt/internal_clean_only_seed0.json \
  --data-root DATA_ROOT --output-root OUTPUT_ROOT \
  --weights WEIGHTS/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth --prepare-only
```

Remove --prepare-only to train. Select yolo, ratio or transfer with its corresponding [configuration](configs). Use one visible GPU and a separate output directory per invocation.

Results: [RT internal](results/recomputed/rt_internal.csv), [RT external](results/recomputed/rt_external.csv), [YOLO external](results/recomputed/yolo_external.csv). Each contains all 32 configurations in fixed subset order, with seeds followed by means. [Klein](../generation/klein/README.md) provides the independent second-editor comparison.
