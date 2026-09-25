# Environments and pretrained weights

Recorded detector environments use Python 3.12.3, PyTorch 2.12.1+cu130 and TorchVision 0.27.1+cu130. Install the matching CUDA wheels separately, then install [requirements-detectors.txt](../requirements-detectors.txt). [environment_versions.json](../configs/environment_versions.json) lists dependency versions. RT source commit is `a21d516aca15da57e65f35c47659c7535ad2b6b3`. YOLO uses Ultralytics 8.3.203 with the included CIoU FP32 patch.

```bash
python -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r downstream/requirements-detectors.txt
mkdir -p weights
curl -L https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth -o weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth
curl -L https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt -o weights/yolo11m.pt
```

The RT initializer SHA-256 is `2ace52184b620204004509b72752ac7bfe64aadaf7fc1d076b18df8ab5a5c77e`. The YOLO initializer SHA-256 is `d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95`. Training verifies these identities. Standard pretrained weights are downloaded separately. The artifacts archive supplies the two experiment-specific frozen references.

Recorded downstream hardware is NVIDIA GeForce RTX 5090. Fits use one visible GPU each; the recorded device capacity is 32607 MiB. TF32, determinism, AMP, loader workers and parameter grouping follow the actual configurations and implementations. YOLO disables TF32; RT uses cuDNN TF32 enabled and matmul TF32 disabled.

The vendored RT implementation retains its license. Ultralytics is governed by its included [license](../licenses/ULTRALYTICS_LICENSE). Install third-party dependencies under their respective licenses.
