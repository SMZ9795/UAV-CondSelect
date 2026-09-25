"""Box-prompt SAM segmentation and source-scale protection masks."""
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

def parse_voc_xml(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    size_node = root.find('size')
    xml_w = int(float(size_node.findtext('width', default='0')))
    xml_h = int(float(size_node.findtext('height', default='0')))
    boxes = []
    for obj in root.findall('object'):
        name = obj.findtext('name', default='object')
        bnd = obj.find('bndbox')
        if bnd is None:
            continue
        xmin = int(float(bnd.findtext('xmin')))
        ymin = int(float(bnd.findtext('ymin')))
        xmax = int(float(bnd.findtext('xmax')))
        ymax = int(float(bnd.findtext('ymax')))
        boxes.append((name, [xmin, ymin, xmax, ymax]))
    if not boxes:
        raise RuntimeError(f'No bbox found in XML: {xml_path}')
    return (xml_w, xml_h, boxes)

def scale_box(box, xml_w, xml_h, img_w, img_h):
    xmin, ymin, xmax, ymax = box
    if xml_w <= 0 or xml_h <= 0:
        return [xmin, ymin, xmax, ymax]
    sx = img_w / xml_w
    sy = img_h / xml_h
    return [int(round(xmin * sx)), int(round(ymin * sy)), int(round(xmax * sx)), int(round(ymax * sy))]

def union_boxes(boxes):
    return [min((b[0] for b in boxes)), min((b[1] for b in boxes)), max((b[2] for b in boxes)), max((b[3] for b in boxes))]

def expand_box(box, img_w, img_h, ratio):
    xmin, ymin, xmax, ymax = box
    bw = xmax - xmin
    bh = ymax - ymin
    px = int(round(bw * ratio))
    py = int(round(bh * ratio))
    return [max(0, xmin - px), max(0, ymin - py), min(img_w, xmax + px), min(img_h, ymax + py)]

def box_to_crop_coords(box, crop_box):
    x1, y1, x2, y2 = box
    cx1, cy1, _, _ = crop_box
    return [x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1]

def resize_crop_and_box(crop_rgb, box_crop, upscale, max_crop_side):
    h, w = crop_rgb.shape[:2]
    scale = upscale
    if max(w, h) * scale > max_crop_side:
        scale = max_crop_side / max(w, h)
    scale = max(1.0, scale)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    if scale != 1.0:
        crop_rgb_resized = cv2.resize(crop_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    else:
        crop_rgb_resized = crop_rgb.copy()
    x1, y1, x2, y2 = box_crop
    box_resized = np.array([x1 * scale, y1 * scale, x2 * scale, y2 * scale], dtype=np.float32)
    return (crop_rgb_resized, box_resized, scale)

def make_box_mask(h, w, box):
    mask = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = box
    x1 = max(0, min(w, int(round(x1))))
    x2 = max(0, min(w, int(round(x2))))
    y1 = max(0, min(h, int(round(y1))))
    y2 = max(0, min(h, int(round(y2))))
    mask[y1:y2, x1:x2] = 255
    return mask

def select_best_mask(masks, scores, bbox_mask_resized):
    best_idx = 0
    best_value = -1000000000.0
    bbox_area = max(1, int((bbox_mask_resized > 0).sum()))
    for i, m in enumerate(masks):
        m_u8 = m.astype(np.uint8) * 255
        inside = ((m_u8 > 0) & (bbox_mask_resized > 0)).sum()
        total = max(1, (m_u8 > 0).sum())
        inside_ratio = inside / total
        area_ratio = inside / bbox_area
        penalty = 0.0
        if area_ratio < 0.08:
            penalty += 0.5
        if area_ratio > 0.95:
            penalty += 0.5
        if inside_ratio < 0.8:
            penalty += 0.5
        value = float(scores[i]) + 0.3 * inside_ratio - penalty
        if value > best_value:
            best_value = value
            best_idx = i
    return best_idx


def load_predictor(checkpoint, model_type):
    import torch
    from segment_anything import sam_model_registry, SamPredictor
    model = sam_model_registry[model_type](checkpoint=str(checkpoint))
    model.to(device='cuda' if torch.cuda.is_available() else 'cpu')
    return SamPredictor(model)


def prepare_masks(image_path, xml_path, predictor, config):
    image = Image.open(image_path).convert('RGB')
    array = np.array(image)
    w, h = image.size
    xml_w, xml_h, objects = parse_voc_xml(xml_path)
    boxes = [scale_box(box, xml_w, xml_h, w, h) for _, box in objects]
    bbox = union_boxes(boxes)
    if not (0 <= bbox[0] < bbox[2] <= w and 0 <= bbox[1] < bbox[3] <= h):
        raise ValueError('Invalid annotation geometry for SAM')
    crop_box = expand_box(bbox, w, h, config['crop_expand'])
    x1, y1, x2, y2 = crop_box
    crop = array[y1:y2, x1:x2].copy()
    crop_h, crop_w = crop.shape[:2]
    prepared, box, scale = resize_crop_and_box(crop, box_to_crop_coords(bbox, crop_box),
                                              config['upscale'], config['max_crop_side'])
    bbox_mask = make_box_mask(*prepared.shape[:2], box)
    predictor.set_image(prepared)
    masks, scores, _ = predictor.predict(point_coords=None, point_labels=None, box=box, multimask_output=True)
    selected = select_best_mask(masks, scores, bbox_mask)
    raw = masks[selected].astype(np.uint8) * 255
    raw_crop = cv2.resize(raw, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
    core = np.zeros((h, w), dtype=np.uint8)
    core[y1:y2, x1:x2] = raw_crop
    if config['constrain_to_bbox']:
        constraint = np.zeros_like(core)
        bx1, by1, bx2, by2 = bbox
        constraint[by1:by2, bx1:bx2] = 255
        core = cv2.bitwise_and(core, constraint)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    cleaned = np.zeros_like(core)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= 5:
            cleaned[labels == i] = 255
    radius = config['protect_px']
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    protection = cv2.dilate(cleaned, kernel, iterations=1)
    return {'raw_selected': Image.fromarray(raw), 'core': Image.fromarray(cleaned),
            'protection': Image.fromarray(protection)}, {'bbox': bbox, 'crop_xyxy': crop_box,
            'actual_scale': scale, 'candidate_scores': [float(s) for s in scores], 'selected': int(selected)}
