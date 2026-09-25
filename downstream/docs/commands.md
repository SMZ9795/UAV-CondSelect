# Response extraction and evaluation

Materialize a configuration before model construction. The preparation command rewrites the selected annotations into the output directory and resolves image paths against the supplied data root.

```bash
python downstream/tools/materialize.py --config downstream/configs/rt/reference_seed0.json \
  --data-root DATA_ROOT --output-root WORK --rt-weights WEIGHTS/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth
python downstream/tools/extract_responses.py --model rt --config WORK/reference_seed0.yml \
  --checkpoint artifacts/references/rt.pth --data-root DATA_ROOT --output WORK/responses.npz
python downstream/tools/evaluate.py --model rt --config WORK/reference_seed0.yml \
  --checkpoint CHECKPOINT --annotation COCO_JSON --data-root DATA_ROOT --output EVALUATION
```

For YOLO, render configs/yolo/reference_seed0.json and use that configuration, `--model yolo`, and `--yolo-weights WEIGHTS/yolo11m.pt`. Evaluate the two external COCO datasets separately and average their AP values equally. Evaluation saves predictions, component metrics. RT loads `ema.module`, applies its 300-query postprocessor and converts original-grid xyxy to xywh. Original size order is width, height. YOLO loads `ema`, uses confidence threshold 0, NMS IoU 0.7, max_nms 30000 and max_det 300, restores stretched coordinates, clamps to the source frame and maps detector class 0 to COCO class 1. RT uses its configured COCO category remapping.

RT context features are output 0 of `encoder` (`HybridEncoder`), shape 1x256x120x120. Protection masks are thresholded at >0, aligned to 960x960 with nearest interpolation, and dilated with a 9x9 square. Context support is the source-box-centered square of side max(4 times the longer box edge, 64), clipped with floor/ceil bounds, excluding the dilated mask. Area interpolation produces fractional feature-grid weights. Object support separately maps the saved binary SAM core mask directly to the feature grid with area interpolation.

YOLO uses the output hook on `model.19.m.0.m.0.cv2`, the transformed internal bottleneck branch of the P4 neck fusion block, shape 1x128x60x60. It uses global support outside the same aligned and dilated protection mask. All six images for a source share weights. Pooling divides the weighted feature sum by support sum plus 1e-12.

The extraction NPZ stores source IDs, clean followed by the fixed five-condition order, pooled context/object features. [recompute.py](../tools/recompute.py) reads the RT and YOLO cache schemas, subtracts paired clean features and normalizes each difference by its L2 norm plus 1e-12. It averages per-source dot products and exports both measured and unit-diagonal selection matrices. RT uses FP32 normalization and reductions; YOLO uses FP64 normalization and reductions after FP32 subtraction.

The shared [selection algorithm](../src/selection.py) starts from all five conditions for every lambda. Coverage is the unnormalized sum over the original five-condition universe; redundancy is the signed sum over unordered selected pairs. Each iteration recomputes all deletion gains, deletes only the greatest strictly positive gain, and resolves exact ties in fixed condition order. Traces include every candidate, F/R/J, the deletion and stopping reason. [select_conditions.py](../tools/select_conditions.py) runs this step directly from an included matrix; the [experiment guide](reproduction.md#select-conditions) gives commands. AP never enters selection.

Transfer gain subtracts the same-seed control AP. Directed gains retain all 20 ordered pairs; symmetric gains average reciprocal directions before averaging seeds. Error bars use sample standard deviation over three seeds. Pearson uses raw values and Spearman uses average ranks.

`convert_annotations.py` provides explicit XML and normalized-YOLO coordinate conversion. Existing generated XML already in 1024x576 must be kept at that grid. Loader COCO IDs, sample pairings and per-image annotation records are supplied under [data](../data).

Generated-image inputs end at final: prepare source and protection mask, blur the target region into an editing input, perform one background edit, then blend the source UAV with its soft mask. Processing ends at final. This downstream module consumes those final images; the independent Klein comparison is under [generation/klein](../../generation/klein/README.md).
