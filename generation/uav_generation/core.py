"""Deterministic preparation, RGB fusion, and annotation geometry."""
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image, ImageFilter


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_image(image, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in ('.jpg', '.jpeg'):
        image.save(path, quality=95)
    else:
        image.save(path)


def resize_source(image, max_side):
    if max_side <= 0:
        return image
    w, h = image.size
    scale = min(1.0, max_side / max(w, h))
    size = (max(16, int(round(w * scale)) // 16 * 16),
            max(16, int(round(h * scale)) // 16 * 16))
    return image if size == image.size else image.resize(size, Image.Resampling.LANCZOS)


def prepare_input(source_path, protection_path, config):
    with Image.open(source_path) as image:
        original_size = image.size
        source = resize_source(image.convert('RGB'), config['max_side'])
    with Image.open(protection_path) as image:
        if image.size != original_size:
            raise ValueError('Protection mask must use the source image coordinate grid')
        hard = image.convert('L').resize(source.size, Image.Resampling.NEAREST)
    hard = hard.point(lambda v: 255 if v > config['mask_threshold'] else 0)
    radius = config['extra_dilate_px']
    if radius > 0:
        hard = hard.filter(ImageFilter.MaxFilter(2 * radius + 1))
        hard = hard.point(lambda v: 255 if v > config['mask_threshold'] else 0)
    soft = hard.filter(ImageFilter.GaussianBlur(config['mask_blur_radius'])) if config['mask_blur_radius'] > 0 else hard
    blurred = source.filter(ImageFilter.GaussianBlur(config['source_blur_radius']))
    editor_input = Image.composite(blurred, source, soft).convert('RGB')
    return source, hard, soft, editor_input


def compose_final(source, background, soft_mask):
    return Image.composite(source, background, soft_mask).convert('RGB')


def convert_annotation(xml_path, size, final_name):
    root = ET.parse(xml_path).getroot()
    width = int(float(root.findtext('size/width', '0')))
    height = int(float(root.findtext('size/height', '0')))
    if width <= 0 or height <= 0:
        raise ValueError('Annotation must declare positive dimensions')
    sx, sy = size[0] / width, size[1] / height
    boxes = root.findall('object/bndbox')
    if not boxes:
        raise ValueError('Annotation contains no boxes')
    for box in boxes:
        values = [int(float(box.findtext(k))) for k in ('xmin', 'ymin', 'xmax', 'ymax')]
        if not (0 <= values[0] < values[2] <= width and 0 <= values[1] < values[3] <= height):
            raise ValueError('Source annotation box is out of bounds')
        values = [int(round(v * s)) for v, s in zip(values, (sx, sy, sx, sy))]
        if not (0 <= values[0] < values[2] <= size[0] and 0 <= values[1] < values[3] <= size[1]):
            raise ValueError('Scaled annotation box is degenerate or out of bounds')
        for k, v in zip(('xmin', 'ymin', 'xmax', 'ymax'), values):
            box.find(k).text = str(v)
    for name, text in [('filename', final_name), ('folder', 'final'), ('path', '../final/' + final_name)]:
        node = root.find(name)
        if node is None:
            node = ET.SubElement(root, name)
        node.text = text
    root.find('size/width').text = str(size[0])
    root.find('size/height').text = str(size[1])
    return ET.ElementTree(root)
