# Downstream experiments

RT-DETRv2-S and YOLO11m training, paired response measurement, condition selection, and evaluation for UAV detection. Conditions follow the order rain, snow, fog, dust, low_light; input records also use dim_low_light for low light.

## Condition selection

Run from the repository root with Python 3.10 or later:

```bash
python downstream/tools/select_conditions.py \
  --matrix downstream/results/recomputed/rt_context_selection.json \
  --lambda 0.6 --output outputs/selection.json
```

The command uses the Python standard library. The [shared algorithm](src/selection.py) evaluates coverage over the full condition pool and signed overlap over retained pairs. It recomputes deletion gains after each step and resolves exact ties in matrix order. [Selection inputs and outputs](results/README.md) describe the matrix format and traces.

## Results and numerical analysis

Install [numerical dependencies](requirements-numerics.txt) to export AP aggregates from the included records:

```bash
python -m pip install -r downstream/requirements-numerics.txt
python downstream/tools/recompute.py --results-only --output outputs/results
```

With the [response caches](data/README.md#response-caches-and-frozen-references) arranged under `artifacts/`, recompute similarities, selection traces, and transfer statistics:

```bash
python downstream/tools/recompute.py --assets artifacts --output outputs/recomputed
python downstream/tools/plot_results.py --results outputs/recomputed --output outputs/figures
```

The [results index](results/README.md) links all included numerical outputs. The [experiment guide](docs/reproduction.md) maps them to the paper.

## Training and evaluation

Follow the [environment and weights guide](docs/environment.md), arrange the [input data](data/README.md), and select an entry from the [configuration index](configs/README.md).

```bash
python downstream/tools/run.py rt --config downstream/configs/rt/internal_clean_only_seed0.json \
  --data-root DATA_ROOT --output-root outputs/rt_clean_seed0 \
  --weights weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth --prepare-only
```

Remove `--prepare-only` to train. Use the `yolo`, `ratio`, or `transfer` family with its corresponding configuration. Each invocation uses one visible GPU and a separate output directory. [Protocols](docs/protocols.md) specify training and metric settings; [commands](docs/commands.md) cover response extraction and evaluation.
