"""Shared five-condition execution; all output paths are relative in manifests."""
import json
import os
import re
from pathlib import Path
from PIL import Image
from .core import prepare_input, compose_final, save_image, sha256, convert_annotation

CONDITIONS = ('rain', 'snow', 'fog', 'dust', 'low_light')


def load_sources(manifest, require_mask=True):
    manifest = Path(manifest).resolve()
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    ids = set()
    if not records:
        raise ValueError('Empty source manifest')
    for record in records:
        sid = record['source_id']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', sid) or sid in ids:
            raise ValueError('Invalid or duplicate source_id: ' + str(sid))
        ids.add(sid)
        keys = ['image', 'annotation'] + (['protection_mask'] if require_mask else [])
        for key in keys:
            value = Path(record[key])
            value = value if value.is_absolute() else manifest.parent / value
            record[key] = value.resolve()
            if not record[key].is_file():
                raise FileNotFoundError(record[key])
        with Image.open(record['image']) as image:
            image.verify()
        if require_mask:
            with Image.open(record['image']) as image, Image.open(record['protection_mask']) as mask:
                if image.size != mask.size:
                    raise ValueError('Source/protection mask dimensions differ')
                mask.verify()
    return records


def validate_config(config, prompts):
    required = {'max_side', 'mask_blur_radius', 'source_blur_radius', 'mask_threshold',
                'extra_dilate_px', 'steps', 'guidance_scale', 'seed', 'dtype', 'cpu_offload', 'conditions'}
    if set(config) != required:
        raise ValueError('Unexpected or missing generation configuration keys')
    conditions = config['conditions']
    if not conditions or len(set(conditions)) != len(conditions) or set(conditions) - set(CONDITIONS):
        raise ValueError('Invalid condition selection')
    import hashlib
    for condition in conditions:
        prompt = prompts[condition]
        if hashlib.sha256(prompt['text'].encode()).hexdigest() != prompt['sha256']:
            raise ValueError('Prompt hash mismatch: ' + condition)
    if config['dtype'] not in ('bf16', 'fp16') or not isinstance(config['cpu_offload'], bool):
        raise ValueError('Invalid precision/offload configuration')
    if config['steps'] < 1 or config['max_side'] < 16 or config['extra_dilate_px'] < 0:
        raise ValueError('Invalid dimensions/steps/dilation')
    if not 0 <= config['mask_threshold'] <= 255 or min(config['mask_blur_radius'], config['source_blur_radius']) < 0:
        raise ValueError('Invalid mask or blur parameters')


def generate_pair(record, condition, config, prompt, editor, output):
    sid = record['source_id']
    suffix = Path(record['image']).suffix.lower()
    if suffix not in ('.jpg', '.jpeg', '.png'):
        raise ValueError('Supported source formats: JPEG and PNG')
    name = f'{sid}_{condition}'
    source, hard, soft, editor_input = prepare_input(record['image'], record['protection_mask'], config)
    annotation = convert_annotation(record['annotation'], source.size, name + suffix)
    paths = {'working_source': f'working_sources/{name}{suffix}',
             'hard_mask': f'masks/{name}_hard.png', 'soft_mask': f'masks/{name}_soft.png',
             'editor_input': f'editor_inputs/{name}{suffix}', 'background': f'backgrounds/{name}{suffix}',
             'final_image': f'final/{name}{suffix}', 'annotation': f'annotations/{name}.xml'}
    for key, image in [('working_source', source), ('hard_mask', hard), ('soft_mask', soft), ('editor_input', editor_input)]:
        save_image(image, output / paths[key])
    background = editor.generate(editor_input, prompt['text'], config)
    # Saving this diagnostic image never changes the in-memory object used below.
    save_image(background, output / paths['background'])
    final_image = compose_final(source, background, soft)
    final_path = output / paths['final_image']
    save_image(final_image, final_path)
    # Only annotation/metadata I/O follows the terminal image save.
    annotation_path = output / paths['annotation']
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation.write(annotation_path, encoding='utf-8', xml_declaration=True)
    return {'source_id': sid, 'condition': condition, 'width': source.width, 'height': source.height,
            'files': paths, 'input_references': {k: os.path.relpath(record[k], output) for k in ('image', 'annotation', 'protection_mask')},
            'input_sha256': {k: sha256(record[k]) for k in ('image', 'annotation', 'protection_mask')},
            'sha256': {k: sha256(output / v) for k, v in paths.items()},
            'prompt_sha256': prompt['sha256'], 'seed': config['seed'], 'background_model_calls': 1,
            'annotation_coordinates': 'raw pixel xyxy; dimensions scaled with round; no origin shift',
            'fusion_input': 'in-memory RGB source and background, shared L soft mask',
            'config': 'config.json', 'prompts': 'prompts.json'}


def run_generation(records, config, prompts, editor, output):
    validate_config(config, prompts)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    (output / 'prompts.json').write_text(json.dumps(prompts, ensure_ascii=False, indent=2))
    with (output / 'manifest.jsonl').open('w') as manifest:
        for record in records:
            for condition in config['conditions']:
                result = generate_pair(record, condition, config, prompts[condition], editor, output)
                manifest.write(json.dumps(result, ensure_ascii=False) + '\n')
                manifest.flush()
    return output
