# ui/utils.py
import cv2
import numpy as np
from typing import List, Tuple

def load_classes(classes_path: str) -> List[str]:
    """
    Read classes.txt with 'index:name' per line.
    Returns list where idx -> name.
    """
    mapping = {}
    with open(classes_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            idx_str, name = line.split(':', 1)
            mapping[int(idx_str.strip())] = name.strip()
    if not mapping:
        raise ValueError("classes.txt is empty or malformed. Expected 'index:name' per line.")
    max_idx = max(mapping.keys())
    return [mapping.get(i, f'class_{i}') for i in range(max_idx + 1)]

def letterbox(im, new_shape=640, color=(114, 114, 114)):
    """
    Resize + pad to meet new_shape while preserving aspect ratio.
    Returns: (img, ratio, (dw, dh))
    """
    shape = im.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def scale_boxes(boxes_xyxy: np.ndarray, ratio: float, pad: Tuple[float, float], orig_shape):
    """
    Map boxes from letterboxed space back to original image coords.
    boxes_xyxy: (N,4) in 640 space; returns clipped copy in original space.
    """
    if boxes_xyxy.size == 0:
        return boxes_xyxy
    dw, dh = pad
    out = boxes_xyxy.copy()
    out[:, [0, 2]] -= dw
    out[:, [1, 3]] -= dh
    out[:, :4] /= ratio
    h, w = orig_shape[:2]
    out[:, 0] = np.clip(out[:, 0], 0, w - 1)
    out[:, 2] = np.clip(out[:, 2], 0, w - 1)
    out[:, 1] = np.clip(out[:, 1], 0, h - 1)
    out[:, 3] = np.clip(out[:, 3], 0, h - 1)
    return out

def _iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (a + b - inter + 1e-7)

def nms(dets: np.ndarray, iou_thres=0.5) -> np.ndarray:
    """
    Class-wise NMS.
    dets: (N,6) -> [x1,y1,x2,y2,score,cls]
    returns filtered dets (M,6)
    """
    if dets is None or len(dets) == 0:
        return np.empty((0, 6), dtype=np.float32)

    out = []
    classes = np.unique(dets[:, 5].astype(int))
    for c in classes:
        dc = dets[dets[:, 5] == c]
        if len(dc) == 0:
            continue
        order = dc[:, 4].argsort()[::-1]
        dc = dc[order]
        keep = []
        while len(dc) > 0:
            i = 0
            keep.append(dc[i])
            if len(dc) == 1:
                break
            ious = _iou(dc[i, :4], dc[1:, :4])
            dc = dc[1:][ious <= iou_thres]
        if keep:
            out.append(np.stack(keep))
    if not out:
        return np.empty((0, 6), dtype=np.float32)
    return np.concatenate(out, axis=0)

def _class_color(cid: int) -> tuple:
    # Simple deterministic palette
    np.random.seed(cid * 97 + 13)
    return tuple(int(x) for x in np.random.randint(64, 255, size=3))

def draw_boxes(frame: np.ndarray, dets: List[tuple], classes: List[str]) -> np.ndarray:
    """
    Draw boxes with label 'name conf%'.
    dets: list of (x1,y1,x2,y2,score,cls_id) in original coords.
    """
    if dets is None:
        return frame
    img = frame.copy()
    for (x1, y1, x2, y2, score, cid) in dets:
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        color = _class_color(int(cid))
        cv2.rectangle(img, p1, p2, color, 2)
        name = classes[int(cid)] if 0 <= int(cid) < len(classes) else f"id{int(cid)}"
        label = f"{name} {int(score*100)}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), color, -1)
        cv2.putText(img, label, (p1[0] + 2, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return img
