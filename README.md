# UAV-CondSelect

Code for the paper "What Should We Generate for UAV Detection?"

## Generation

The [generation module](generation/README.md) prepares protection masks, edits the background with FLUX.2-dev, and integrates the source UAV to produce the final image. It supports rain, snow, fog, dust, and low light.

See the [installation instructions](generation/ENVIRONMENT.md), [data format](generation/DATA.md), and [example files](generation/examples/README.md). Run the documented commands from `generation/`.

### Complete example

The fog example includes the source image, annotations, masks, editor input, generated background, and final output.

![Fog generation workflow](generation/examples/workflow.png)

### Selected qualitative examples

Source and final images for rain, snow, fog, and dust. The [example manifest](generation/examples/gallery/manifest.json) records their annotations, settings, and checksums.

![Selected qualitative examples](generation/examples/gallery/overview.png)
