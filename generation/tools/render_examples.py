#!/usr/bin/env python3
"""Render the generation workflow from its example files."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONT = ImageFont.load_default(size=20)
SMALL = ImageFont.load_default(size=18)
W, H, LABEL, GAP = 512, 288, 32, 12


def panel(canvas, xy, path, label, binary=False):
    x, y = xy
    ImageDraw.Draw(canvas).text((x + 6, y + 5), label, fill='#172638', font=SMALL)
    with Image.open(path) as im:
        im = im.convert('RGB')
        im.thumbnail((W, H), Image.Resampling.NEAREST if binary else Image.Resampling.LANCZOS)
        canvas.paste(im, (x + (W-im.width)//2, y + LABEL + (H-im.height)//2))


def main():
    ex = ROOT / 'examples'
    flow = json.loads((ex/'flow_manifest.json').read_text())
    stages = [('source_image', 'Source'), ('core_mask', 'SAM core mask'),
              ('protection_mask', 'Protection mask'), ('working_source', 'Working source'),
              ('hard_mask', 'Hard mask'), ('soft_mask', 'Soft mask'),
              ('editor_input', 'Editor input'), ('background', 'Generated background'),
              ('final_image', 'Final')]
    canvas = Image.new('RGB', (3*W+4*GAP, 3*(H+LABEL)+4*GAP+40), '#f2f4f7')
    ImageDraw.Draw(canvas).text((GAP, 12), 'Fog | A_0106837 | Generation workflow', fill='#172638', font=FONT)
    for i, (key, label) in enumerate(stages):
        panel(canvas, (GAP+(i%3)*(W+GAP), 40+GAP+(i//3)*(H+LABEL+GAP)),
              ex/flow['files'][key]['path'], label, key in ('core_mask', 'protection_mask', 'hard_mask'))
    canvas.save(ex/'workflow.png')


if __name__ == '__main__':
    main()
