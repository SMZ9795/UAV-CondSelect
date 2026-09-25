# Data layout

Training and extraction resolve image and mask paths against the `--data-root` directory. Keep source IDs, file names, and the supplied split membership when arranging data.

## Images and annotations

| File | Purpose |
| --- | --- |
| [image_manifest.csv](image_manifest.csv) | Main training and evaluation images, dimensions, source IDs, splits, and annotation references |
| [response_pairs.csv](response_pairs.csv) | Paired clean/generated images, boxes, core masks, and protection masks for response measurement |
| [transfer_image_manifest.csv](transfer_image_manifest.csv) | Image paths and annotation references for transfer experiments |
| [loader_annotations](loader_annotations) | COCO annotations used by the supplied configurations |
| [splits](splits) | Source-level membership shared by each source and its generated variants |

For each main image, place the file at `DATA_ROOT/<image_relative_path>` from `image_manifest.csv`. Transfer uses its `path` column. For response measurement, use the four path columns `clean_path`, `weather_path`, `core_mask_path`, and `protect_mask_path` in `response_pairs.csv`. Preserve the dimensions specified by each record.

The main clean split contains 511 training, 110 validation, and 109 test images. Each generated condition contributes 102, 22, and 22 images respectively. External training uses 730 clean images and 146 images per retained condition. The frozen reference uses 409 extra-clean training sources; response measurement uses the disjoint 102 generation sources. See [protocols](../docs/protocols.md).

[run.py](../tools/run.py) materializes the selected COCO annotation files with absolute image paths under the chosen output directory. Generated XML already in the 1024x576 grid keeps that grid. [Generation data formats](../../generation/DATA.md) describe image generation and annotation scaling.

Transfer training also reads `train_sources.txt`, `val_sources.txt`, and `test_sources.txt` from `DATA_ROOT/rtdetrv2_pilot_dataset/splits/`. Copy these three files from [splits](splits).

## Response caches and frozen references

Numerical recomputation accepts an artifact directory containing `responses/rt.npz` and `responses/yolo.npz`. Response extraction uses `references/rt.pth` and `references/yolo.pt` as the frozen detector checkpoints; [extraction commands](../docs/commands.md) describe how to produce the caches. Standard pretrained initializers have separate [download instructions](../docs/environment.md).

The mask installation command reads `masks/core_mask_path/<source_id>.png` and `masks/protect_mask_path/<source_id>.png` from the artifact directory and places them at the paths in `response_pairs.csv`:

```bash
python downstream/tools/install_masks.py --assets artifacts --data-root DATA_ROOT
```

The [Klein records](../../generation/klein/records.json) resolve their asset paths under `artifacts/klein`. The [Klein guide](../../generation/klein/README.md) specifies preparation and generation commands. Included [selection matrices and AP records](../results/README.md) can be used directly from the repository.
