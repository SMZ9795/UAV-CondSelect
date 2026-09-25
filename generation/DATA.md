# Data and processing

## Input manifest

Use one JSON object per line, with paths relative to the manifest directory:

```json
{"source_id":"A_0106837","image":"images/A_0106837.jpg","annotation":"annotations/A_0106837.xml","protection_mask":"masks/A_0106837_protection.png"}
```

Each `source_id` must be unique and contain letters, digits, underscores, or hyphens. For mask preparation, supply `source_id`, `image`, and `annotation`. Generation also requires a `protection_mask` matching the source image dimensions. The optional `core_mask` field records the target mask.

Annotations use VOC-style XML with image dimensions and pixel XYXY boxes. File pairing follows the manifest. Coordinates are read with `int(float(value))`, scaled independently in x and y from the XML grid to the output grid, and rounded to integers. Boxes must have positive area and lie within the declared image dimensions.

## Masks

[configs/sam.json](configs/sam.json) defines the SAM settings. The default `vit_b` model uses the enclosing box of all annotated targets, a context crop expanded by 0.8, upscaling by 4, and a maximum crop side of 1024. Candidate selection combines the SAM score, the fraction inside the box, and area penalties. Connected components smaller than five pixels are removed to form the core mask.

The protection mask is the core dilated once with a 9 × 9 elliptical kernel. It is shared across conditions for each source image. Mask preparation saves the selected candidate, core, protection mask, and selection metadata.

## Generation

[configs/generation.json](configs/generation.json) defines the processing settings:

| Step | Default |
| --- | --- |
| Working image | Longest side capped at 1024; dimensions rounded down to multiples of 16; LANCZOS resizing |
| Hard mask | Protection mask resized with NEAREST, then thresholded at 127 |
| Soft mask | Pillow GaussianBlur, radius 28 |
| Editor input | Source blurred at radius 60 in the region weighted by the soft mask |
| Background | FLUX.2-dev, BF16, CPU offload, 28 steps, guidance 3.0, seed 43 per image-condition pair |
| Final | Source and generated background composited in memory with the soft mask |

Fusion uses 8-bit RGB values and the L-mode soft mask. The background is resized to the working grid with LANCZOS when needed. PNG sources produce PNG outputs; JPEG outputs use quality 95. Prompt text and UTF-8 SHA-256 values are stored in [prompts/background.json](prompts/background.json); update both when editing a prompt.

## Output records

Each output record identifies its source, condition, settings, annotations, and image files. Paths are relative to the output directory, and files include SHA-256 checksums. The final image is recorded as `files.final_image`.

Example files are indexed in [examples/README.md](examples/README.md). The [flow manifest](examples/flow_manifest.json) describes the complete fog example; the [gallery manifest](examples/gallery/manifest.json) pairs each condition's source and final image.
