#!/usr/bin/env python3
"""Unified entry point for mask preparation and final-image generation."""
import argparse
import json
import os
from pathlib import Path
from uav_generation.workflow import load_sources, validate_config, run_generation

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare-masks', 'generate'):
        sub = commands.add_parser(name)
        sub.add_argument('--manifest', type=Path, required=True)
        sub.add_argument('--output', type=Path, required=True, help='New output directory; existing paths are refused')
        sub.add_argument('--config', type=Path, default=ROOT / 'configs' / ('sam.json' if name == 'prepare-masks' else 'generation.json'))
        if name == 'generate':
            sub.add_argument('--model', type=Path, required=True)
            sub.add_argument('--prompts', type=Path, default=ROOT / 'prompts/background.json')
            sub.add_argument('--conditions', nargs='+', help='Subset of configured conditions')
            sub.add_argument('--dry-run', action='store_true')
        else:
            sub.add_argument('--sam-checkpoint', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new directory')
    config = json.loads(args.config.read_text())
    records = load_sources(args.manifest, require_mask=args.command == 'generate')
    if args.command == 'generate':
        if args.conditions:
            config['conditions'] = args.conditions
        prompts = json.loads(args.prompts.read_text())
        validate_config(config, prompts)
        # CPU preflight validates geometry before loading a large model.
        from uav_generation.core import prepare_input, convert_annotation
        for record in records:
            source, _, _, _ = prepare_input(record['image'], record['protection_mask'], config)
            convert_annotation(record['annotation'], source.size, record['source_id'] + '.jpg')
        if args.dry_run:
            print(json.dumps({'source_count': len(records), 'conditions': config['conditions'],
                              'background_model_calls_planned': len(records) * len(config['conditions']),
                              'model_inference_executed': False}, indent=2))
            return
        if not args.model.is_dir():
            parser.error('Local FLUX model directory does not exist')
        os.environ.setdefault('OMP_NUM_THREADS', '8')
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
        from uav_generation.editor import FluxBackground
        editor = FluxBackground(args.model, config)
        run_generation(records, config, prompts, editor, args.output)
    else:
        if not args.sam_checkpoint.is_file():
            parser.error('SAM checkpoint does not exist')
        from uav_generation.sam_masks import load_predictor, prepare_masks
        predictor = load_predictor(args.sam_checkpoint, config['model_type'])
        args.output.mkdir(parents=True, exist_ok=False)
        out = args.output.resolve()
        (out / 'config.json').write_text(json.dumps(config, indent=2))
        with (out / 'sources.jsonl').open('w') as manifest:
            for record in records:
                images, metadata = prepare_masks(record['image'], record['annotation'], predictor, config)
                sid = record['source_id']
                for key, image in images.items():
                    image.save(out / f'{sid}_{key}.png')
                (out / f'{sid}_sam.json').write_text(json.dumps(metadata, indent=2))
                manifest.write(json.dumps({'source_id': sid,
                    'image': os.path.relpath(record['image'], out),
                    'annotation': os.path.relpath(record['annotation'], out),
                    'core_mask': f'{sid}_core.png', 'protection_mask': f'{sid}_protection.png'}) + '\n')


if __name__ == '__main__':
    main()
