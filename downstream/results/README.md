# Results

All paths below are relative to this directory. R, S, F, D, and L denote rain, snow, fog, dust, and low light. Main AP tables use Clean-only followed by subsets of size one through five, with combinations in R/S/F/D/L order. Each subset lists seeds 0, 1, and 2 followed by its mean.

## Detection AP

| Experiment | Per-seed results and means |
| --- | --- |
| RT-DETRv2-S, internal | [rt_internal.csv](recomputed/rt_internal.csv) |
| RT-DETRv2-S, external | [rt_external.csv](recomputed/rt_external.csv) |
| YOLO11m, external | [yolo_external.csv](recomputed/yolo_external.csv) |
| Full generation with matched sampling | [per seed](recomputed/ratio_per_seed.csv), [means](recomputed/ratio_summary.csv) |

AP is expressed in percentage points. Internal AP averages clean and five condition-specific evaluations equally. External AP is the equal-weight mean of pooled-real and synthetic evaluations. Means and sample standard deviations are computed across three seeds after per-run aggregation. [runs.json](runs.json) and [ratio_runs.json](ratio_runs.json) contain component AP values; [recompute.py](../tools/recompute.py) exports the aggregates.

## Response similarities and selection

| Detector | Measured similarities | Selection input | Selection traces |
| --- | --- | --- | --- |
| RT-DETRv2-S | [context](recomputed/rt_context_measured.json) | [matrix](recomputed/rt_context_selection.json) | [traces](recomputed/rt_context_selection_traces.json) |
| YOLO11m | [context](recomputed/yolo_context_measured.json) | [matrix](recomputed/yolo_context_selection.json) | [traces](recomputed/yolo_context_selection_traces.json) |

The measured files contain average cosine similarities of paired responses. Selection matrices have a unit diagonal. Their JSON fields are `order` (condition names) and `matrix` (a symmetric square array in that order). Negative similarities retain their sign.

[select_conditions.py](../tools/select_conditions.py) takes a matrix and `--lambda`. The JSON output records the retained conditions, deletion sequence, candidate gains, and final objective values. `F` is coverage over the full input condition pool, `R` is the signed sum over unordered retained pairs, and `J = F - lambda * R`. Every run starts from all input conditions.

## Cross-condition transfer

| Output | Files |
| --- | --- |
| Directed gains | [per seed](recomputed/transfer_directed_per_seed.csv), [means](recomputed/transfer_directed_mean.csv) |
| Symmetric gains | [per seed](recomputed/transfer_symmetric_per_seed.csv), [means](recomputed/transfer_symmetric_mean.csv) |
| Response-transfer correlations | [correlations](recomputed/transfer_correlations.csv) |
| Object-region response comparison | [measured matrix](recomputed/rt_object_measured.json) |

[transfer_AP.json](transfer_AP.json) stores the component AP values on the 0-1 scale. Transfer gains convert these to AP points and subtract the same-seed clean-repeat control. Symmetric gains average reciprocal directions. The [experiment guide](../docs/reproduction.md) gives the corresponding configurations and commands.
