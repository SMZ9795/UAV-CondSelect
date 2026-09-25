#!/usr/bin/env python3
"""CPU checks for input preparation, compositing and output records."""
import argparse
import ast
import contextlib
import hashlib
from fractions import Fraction
import io
import json
import sys
import tempfile
import types
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uav_generation.core import prepare_input, compose_final, convert_annotation, sha256
from uav_generation.workflow import load_sources, run_generation
from uav_generation.editor import FluxBackground


def jpeg_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=95)
    return buffer.getvalue()


def validate():
    config = json.loads((ROOT / 'configs/generation.json').read_text())
    prompts = json.loads((ROOT / 'prompts/background.json').read_text())
    record = load_sources(ROOT / 'examples/sources.jsonl')[0]
    source, hard, soft, editor_input = prepare_input(record['image'], record['protection_mask'], config)
    ref = ROOT / 'examples/reference'
    reference_checks = {
        'working_source_jpeg': jpeg_bytes(source) == (ref / 'working_source.jpg').read_bytes(),
        'hard_mask_pixels': hard.tobytes() == Image.open(ref / 'hard.png').tobytes(),
        'soft_mask_pixels': soft.tobytes() == Image.open(ref / 'soft.png').tobytes(),
        'editor_input_jpeg': jpeg_bytes(editor_input) == (ref / 'editor_input.jpg').read_bytes(),
    }
    assert all(reference_checks.values()), reference_checks
    # Independent integer oracle tests real orchestration + real editor.generate through a fake pipe.
    yy, xx = np.indices((source.height, source.width))
    background = Image.fromarray(np.stack([(xx * 37 + yy * 19) % 256,
        (xx * 11 + yy * 53) % 256, (xx * 97 + yy * 7) % 256], axis=-1).astype('uint8'))
    oracle_pixels = ((np.array(source).astype('uint32') * np.array(soft)[:, :, None]
                     + np.array(background).astype('uint32') * (255 - np.array(soft)[:, :, None]) + 127) // 255).astype('uint8')
    expected_final = jpeg_bytes(Image.fromarray(oracle_pixels))
    lossy_background = Image.open(io.BytesIO(jpeg_bytes(background))).convert('RGB')
    assert jpeg_bytes(compose_final(source, lossy_background, soft)) != expected_final
    calls = []
    class Generator:
        def __init__(self, device): self.device = device
        def manual_seed(self, seed): self.seed = seed; return self
    def pipe(**kwargs):
        assert kwargs['image'].tobytes() == editor_input.tobytes()
        calls.append({'prompt': kwargs['prompt'], 'steps': kwargs['num_inference_steps'],
                      'guidance': kwargs['guidance_scale'], 'seed': kwargs['generator'].seed})
        return types.SimpleNamespace(images=[background.copy()])
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False),
                                       Generator=Generator, inference_mode=contextlib.nullcontext)
    editor = FluxBackground.__new__(FluxBackground)
    editor.pipe = pipe
    with tempfile.TemporaryDirectory(prefix='uav-final-validation-') as tmp:
        output = Path(tmp) / 'results'
        with patch.dict(sys.modules, {'torch': fake_torch}):
            run_generation([record], config, prompts, editor, output)
        rows = [json.loads(line) for line in (output / 'manifest.jsonl').read_text().splitlines()]
        assert len(calls) == len(rows) == 5
        for row, call in zip(rows, calls):
            assert call == {'prompt': prompts[row['condition']]['text'], 'steps': 28, 'guidance': 3.0, 'seed': 43}
            assert (output / row['files']['final_image']).read_bytes() == expected_final
            for key, name in row['files'].items():
                assert sha256(output / name) == row['sha256'][key]
            annotation = ET.parse(output / row['files']['annotation']).getroot()
            assert [int(annotation.findtext('size/' + k)) for k in ('width', 'height')] == [1024, 576]
            input_xml = ET.parse(record['annotation']).getroot()
            for old_box, new_box in zip(input_xml.findall('object/bndbox'), annotation.findall('object/bndbox')):
                for key, dim, length in [('xmin', 'width', 1024), ('xmax', 'width', 1024),
                                         ('ymin', 'height', 576), ('ymax', 'height', 576)]:
                    expected_coordinate = round(Fraction(int(float(old_box.findtext(key))) * length,
                                                        int(input_xml.findtext('size/' + dim))))
                    assert int(new_box.findtext(key)) == expected_coordinate
        # Output overwrites are rejected.
        try:
            run_generation([record], config, prompts, editor, output)
        except FileExistsError:
            pass
        else:
            raise AssertionError('Existing output directory accepted')
        assert len(calls) == 5
    model_calls = []
    for p in [ROOT / 'generate.py', *sorted((ROOT / 'uav_generation').glob('*.py'))]:
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and ast.unparse(node.func) == 'self.pipe':
                model_calls.append({'file': str(p.relative_to(ROOT)), 'line': node.lineno})
    assert len(model_calls) == 1
    # Check source-level terminal image write within the production per-pair function.
    tree = ast.parse((ROOT / 'uav_generation/workflow.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'generate_pair')
    terminal = next(i for i, n in enumerate(function.body) if isinstance(n, ast.Expr)
                    and isinstance(n.value, ast.Call) and ast.unparse(n.value) == 'save_image(final_image, final_path)')
    subsequent = ast.Module(body=function.body[terminal + 1:], type_ignores=[])
    assert not any(isinstance(n, ast.Call) and ast.unparse(n.func) in
                   ['save_image', 'compose_final', 'editor.generate'] for n in ast.walk(subsequent))
    # Check source-scale core/protection relation, independent of model inference.
    import cv2
    core = np.array(Image.open(ROOT / 'examples/input/core.png'))
    protection = np.array(Image.open(ROOT / 'examples/input/protection.png'))
    assert np.array_equal(cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))), protection)
    return {'static_checks': {'single_background_pipeline_call_site': model_calls,
            'terminal_final_save': True, 'syntax': 'passed'},
            'cpu_reference_checks': reference_checks,
            'five_condition_stub_integration': {'calls': 5, 'calls_per_pair': 1,
                'in_memory_fusion_vs_independent_integer_oracle': 'passed',
                'lossy_background_reload_regression_guard': 'passed',
                'annotations_and_manifest_hashes': 'passed', 'existing_output_refusal': 'passed'},
            'source_mask_dilation_check': 'passed',
            'examples': __import__('check_examples').check_examples()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = validate()
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.report:
        with args.report.open('x') as f: f.write(text)
    print(text)
