# Experiment guide

Run commands from the repository root. [Configurations](../configs/README.md) map each condition subset to its training files. [Protocols](protocols.md) define splits, training budgets, sampling, and AP aggregation.

## Paper-to-code index

| Paper experiment | Configuration or input | Result files |
| --- | --- | --- |
| RT response similarities and selection: Figure 4, Table 1 | [frozen reference](../configs/rt/reference_seed0.json), [selection matrix](../results/recomputed/rt_context_selection.json) | [measured similarities](../results/recomputed/rt_context_measured.json), [selection traces](../results/recomputed/rt_context_selection_traces.json) |
| RT detection: Tables 2 and 5 | [RT configurations](../configs/README.md#main-detection-experiments) | [internal AP](../results/recomputed/rt_internal.csv), [external AP](../results/recomputed/rt_external.csv) |
| YOLO response similarities and selection: Figure 5, Table 3 | [frozen reference](../configs/yolo/reference_seed0.json), [selection matrix](../results/recomputed/yolo_context_selection.json) | [measured similarities](../results/recomputed/yolo_context_measured.json), [selection traces](../results/recomputed/yolo_context_selection_traces.json) |
| YOLO detection: Table 4 | [YOLO configurations](../configs/README.md#main-detection-experiments) | [external AP](../results/recomputed/yolo_external.csv) |
| Matched sampling: Table 6 | [ratio configurations](../configs/README.md#matched-sampling) | [per-seed AP](../results/recomputed/ratio_per_seed.csv), [means](../results/recomputed/ratio_summary.csv) |
| Cross-condition transfer: Figure 6, Tables 7 and 8 | [transfer configurations](../configs/README.md#cross-condition-transfer) | [symmetric gains](../results/recomputed/transfer_symmetric_mean.csv), [directed gains](../results/recomputed/transfer_directed_mean.csv), [correlations](../results/recomputed/transfer_correlations.csv) |
| Second-editor illustration: Appendix E, Table 9 | [Klein configuration](../../generation/klein/config.json), [pair records](../../generation/klein/records.json) | [Klein example files](../../generation/klein/examples) |

## Select conditions

Use Python 3.10 or later. The standalone selector uses only the standard library:

```bash
python downstream/tools/select_conditions.py \
  --matrix downstream/results/recomputed/rt_context_selection.json \
  --lambda 0.6 --output outputs/rt_selection.json
python downstream/tools/select_conditions.py \
  --matrix downstream/results/recomputed/yolo_context_selection.json \
  --lambda 0.3 --output outputs/yolo_selection.json
```

The outputs retain snow/dust/low_light and rain/snow/dust respectively. Table 1 uses lambda values 0, 0.3, 0.6, 0.8, 2, 4, and 6; Table 3 uses 0, 0.1, 0.3, and 1. Each invocation starts from the full condition pool.

## Recompute numerical results

```bash
python -m pip install -r downstream/requirements-numerics.txt
python downstream/tools/recompute.py --results-only --output outputs/results
```

This exports AP aggregates from the included per-run records. To recompute response matrices, selection traces, and transfer correlations, arrange the [response caches](../data/README.md#response-caches-and-frozen-references) and run:

```bash
python downstream/tools/recompute.py --assets artifacts --output outputs/recomputed
python downstream/tools/plot_results.py --results outputs/recomputed --output outputs/figures
```

## Train and evaluate detectors

Install the [detector environment and initializers](environment.md) and arrange [images and annotations](../data/README.md). For example, train the RT external three-condition configuration:

```bash
python downstream/tools/run.py rt \
  --config downstream/configs/rt/external_snow_dust_dim_seed0.json \
  --data-root DATA_ROOT --output-root outputs/rt_sdl_seed0 \
  --weights weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth
```

Add `--prepare-only` to materialize configuration and annotations. Repeat with seed1 and seed2 configuration files for the three-seed mean. Use the configuration index to select other subsets and experiment families. [Response extraction and evaluation](commands.md) describes frozen-reference measurement and COCO evaluation.

## Generate paired images

Follow the [generation guide](../../generation/README.md) for mask preparation, background editing, and source-UAV blending. The [examples index](../../generation/examples/README.md) links a complete workflow and condition-specific outputs. The [Klein guide](../../generation/klein/README.md) covers the independent second-editor comparison.
