# Klein Base second-editor comparison

FLUX.2-klein-base-9B uses no task fine-tuning or learned adapter. The [records](records.json) contain 22 validation source IDs, four conditions and 88 pairs, including prompts, inputs, masks, backgrounds, final files and crop coordinates. Relative asset paths refer to artifacts/klein from the separate assets archive.

Use Python 3.12, PyTorch 2.12.1+cu130, Pillow 12.0.0, Transformers 5.14.1, Accelerate 1.14.0, Safetensors 0.8.0 and Diffusers commit 6ad357395d936c4d27347463f938cdbb400a6e59. Model revision is 32773329fbe7e81a90ef971740e8ba4b0364ecf3. See the [model license](MODEL_LICENSE.md) and [configuration](config.json).

```bash
pip install git+https://github.com/huggingface/diffusers.git@6ad357395d936c4d27347463f938cdbb400a6e59
hf download black-forest-labs/FLUX.2-klein-base-9B --revision 32773329fbe7e81a90ef971740e8ba4b0364ecf3 --local-dir MODEL_DIR
python generation/klein/klein.py prepare --source SOURCE --protection MASK --output prepared
python generation/klein/klein.py generate --assets artifacts/klein --model MODEL_DIR \
  --revision 32773329fbe7e81a90ef971740e8ba4b0364ecf3 --output outputs
python generation/klein/klein.py blend --source SOURCE_JPEG --mask SOFT_MASK \
  --background BACKGROUND --output final.png
```

Preparation uses 1024x576, protection threshold >127 without dilation, Gaussian mask radius 28 and source blur radius 60, then saves quality-95 JPEG. Generation decodes the input JPEG and resizes it with Lanczos to 512x288. BF16 inference performs one background edit with 28 steps, guidance 3 and seed 43. It saves background.png and blends the source with the soft mask into final.png, where processing ends.

Both dev and Klein backgrounds use the same decoded source JPEG and soft mask at the independent 512x288 comparison grid. Images use Lanczos and masks use NEAREST. Dev backgrounds are saved at 1024x576; Klein backgrounds at 512x288. This qualitative comparison has no detection-performance results.
