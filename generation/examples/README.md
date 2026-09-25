# Examples

## Generation workflow

The fog example uses source `A_0106837`. The [workflow image](workflow.png) shows the source, intermediate stages, and final output. File metadata and settings are in [flow_manifest.json](flow_manifest.json).

| Stage | Files |
| --- | --- |
| Source | [image](input/source.jpg), [annotation](input/source.xml) |
| Core mask | [core.png](input/core.png) |
| Protection mask | [protection.png](input/protection.png) |
| Working source | [working_source.jpg](reference/working_source.jpg) |
| Hard mask | [hard.png](reference/hard.png) |
| Soft mask | [soft.png](reference/soft.png) |
| Editor input | [editor_input.jpg](reference/editor_input.jpg) |
| Generated background | [background.jpg](reference/background.jpg) |
| Final | [image](reference/final.jpg), [annotation](reference/final.xml) |

Rebuild the workflow image with `python tools/render_examples.py` from `generation/`.

## Condition examples

| Condition | Source ID | Source | Final |
| --- | --- | --- | --- |
| Rain | F_0102954 | [image](gallery/rain/source.jpg), [annotation](gallery/rain/source.xml) | [image](gallery/rain/final.jpg), [annotation](gallery/rain/final.xml) |
| Snow | A_0106838 | [image](gallery/snow/source.jpg), [annotation](gallery/snow/source.xml) | [image](gallery/snow/final.jpg), [annotation](gallery/snow/final.xml) |
| Fog | A_0106837 | [image](gallery/fog/source.jpg), [annotation](gallery/fog/source.xml) | [image](gallery/fog/final.jpg), [annotation](gallery/fog/final.xml) |
| Dust | H_0103862 | [image](gallery/dust/source.jpg), [annotation](gallery/dust/source.xml) | [image](gallery/dust/final.jpg), [annotation](gallery/dust/final.xml) |

[gallery/manifest.json](gallery/manifest.json) records file dimensions, checksums, configuration, and prompt references.
