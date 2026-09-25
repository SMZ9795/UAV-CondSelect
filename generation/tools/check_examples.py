"""Check example files, hashes, image geometry, annotations and local links."""
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def check_examples():
    checked = set()
    def file_entry(base, entry):
        path = (base / entry['path']).resolve()
        assert path.is_relative_to(ROOT) and path.is_file(), path
        assert path.stat().st_size == entry['bytes'], path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], path
        if 'width' in entry:
            with Image.open(path) as im:
                im.load()
                assert im.size == (entry['width'], entry['height']), path
                assert im.mode == entry['mode'], path
        checked.add(path)
        return path
    def annotation(path, image):
        xml = ET.parse(path).getroot()
        with Image.open(image) as im: size = im.size
        assert tuple(int(xml.findtext('size/'+k)) for k in ('width','height')) == size
        boxes = []
        for box in xml.findall('object/bndbox'):
            x0, y0, x1, y1 = [int(float(box.findtext(k))) for k in ('xmin','ymin','xmax','ymax')]
            assert 0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1], path
            boxes.append((x0,y0,x1,y1))
        assert boxes, path
        return size, boxes
    manifests = [ROOT/'examples/flow_manifest.json', ROOT/'examples/gallery/manifest.json']
    count = 0
    for manifest in manifests:
        data = json.loads(manifest.read_text())
        rows = data.get('examples', [data])
        if 'examples' in data:
            assert {r['condition'] for r in rows} == {'rain','snow','fog','dust'}
            assert len(rows) == 4
        for row in rows:
            files = {key:file_entry(manifest.parent, value) for key,value in row['files'].items()}
            config = json.loads(file_entry(manifest.parent,row['config']).read_text())
            prompt = json.loads(file_entry(manifest.parent,row['prompt']).read_text())
            assert row['condition'] in config['conditions']
            assert row['prompt']['condition_key'] == row['condition']
            assert hashlib.sha256(prompt[row['condition']]['text'].encode()).hexdigest() == row['prompt']['text_sha256']
            source_size, source_boxes = annotation(files['source_annotation'],files['source_image'])
            final_size, final_boxes = annotation(files['final_annotation'],files['final_image'])
            assert len(source_boxes) == len(final_boxes)
            for src, dst in zip(source_boxes,final_boxes):
                assert tuple(round(v*final_size[i%2]/source_size[i%2]) for i,v in enumerate(src)) == dst
            for key in ('core_mask','protection_mask'):
                if key in files:
                    with Image.open(files[key]) as im:
                        assert im.size == source_size and set(im.getdata()) <= {0,255}
            for key in ('working_source','hard_mask','soft_mask','editor_input','background'):
                if key in files:
                    with Image.open(files[key]) as im: assert im.size == final_size
            count += 1
    reference = json.loads((ROOT/'examples/reference_manifest.json').read_text())
    for key in ('source','annotation','working_source','hard_mask','soft_mask','editor_input','background','final_image'):
        path = ROOT/'examples'/reference[key]
        assert path.is_file(), path
        if path.name in reference['sha256']:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == reference['sha256'][path.name]
    with Image.open(ROOT/'examples/workflow.png') as im:
        im.load()
    links = 0
    for doc in ROOT.rglob('*.md'):
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', doc.read_text()):
            if '://' in target or target.startswith('#'): continue
            assert (doc.parent/target.split('#')[0]).exists(), (doc,target)
            links += 1
    return {'example_records':count,'unique_manifest_files':len(checked),
            'local_markdown_links':links,'image_annotation_hash_prompt_checks':'passed'}


if __name__ == '__main__':
    print(json.dumps(check_examples(),indent=2))
