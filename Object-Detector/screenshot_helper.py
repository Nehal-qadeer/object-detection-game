# screenshot_helper.py
from pathlib import Path
import cv2, re
from datetime import datetime

def _stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

def _safe(s: str):
    return re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", s)

def save_best_detection(frame_bgr, dets, class_names, out_dir: Path):
    """
    dets: list of (x1,y1,x2,y2,cid,conf)
    Saves CROP + FRAME-with-box for the highest-confidence detection.
    Returns (crop_path, frame_path, cls_id) or (None, None, None) if no detections.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if not dets:
        return None, None, None

    # pick highest-confidence detection
    best = max(dets, key=lambda d: d[5])
    x1,y1,x2,y2,cid,conf = best

    h,w = frame_bgr.shape[:2]
    x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w-1))
    y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h-1))
    if x2 <= x1 or y2 <= y1:
        return None, None, None

    cls_name = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
    tag = f"{_safe(cls_name)}_{_stamp()}_conf{conf:.2f}"

    crop = frame_bgr[y1:y2, x1:x2].copy()
    crop_path = out_dir / f"detected_{tag}_crop.jpg"
    cv2.imwrite(str(crop_path), crop)

    drawn = frame_bgr.copy()
    cv2.rectangle(drawn, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(drawn, f"{cls_name} {conf:.2f}", (x1, max(0, y1-6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
    frame_path = out_dir / f"detected_{tag}_frame.jpg"
    cv2.imwrite(str(frame_path), drawn)

    return str(crop_path), str(frame_path), int(cid)
