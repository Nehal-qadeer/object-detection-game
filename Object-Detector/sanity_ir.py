import time
from pathlib import Path
import cv2
import numpy as np
import openvino as ov

# NEW: helpers (uncommented as requested)
from screenshot_helper import save_best_detection
from quiz_helper import run_quiz

ROOT = Path(__file__).resolve().parent
MODEL_XML = ROOT / "models" / "model_openvino.xml"
MODEL_BIN = ROOT / "models" / "model_openvino.bin"
CLASSES_TXT = ROOT / "models" / "classes.txt"
DEVICE = "CPU"
IMG_SIZE = 640

CONF_THRES = 0.65
IOU_THRES = 0.45
MAX_DETS = 100
PRE_NMS_TOPK = 300
USE_SOFTMAX = False

# -------------------- Subtitles (built-in, no extra files) --------------------
class SubtitleBar:
    def __init__(self):
        self.text = ""
        self.expires_at = 0.0

    def show(self, text: str, secs: float = 2.5):
        self.text = text or ""
        self.expires_at = time.time() + max(0.1, secs)

    def draw(self, frame):
        if not self.text or time.time() > self.expires_at:
            return
        h, w = frame.shape[:2]
        banner_h = max(40, int(0.11 * h))
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - banner_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        x = 12
        y = h - banner_h + 32
        cv2.putText(
            frame, self.text, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA
        )

subs = SubtitleBar()
# -----------------------------------------------------------------------------


def load_names(p: Path):
    if not p.exists():
        return [f"cls{i}" for i in range(10)]
    with open(p, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-9)


def letterbox(im, new_shape=640, color=(114, 114, 114)):
    h, w = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw //= 2
    dh //= 2
    if (w, h) != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(
        im,
        dh,
        new_shape[0] - new_unpad[1] - dh,
        dw,
        new_shape[1] - new_unpad[0] - dw,
        cv2.BORDER_CONSTANT,
        value=color,
    )
    return im, r, (dw, dh)


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    r, (dw, dh) = ratio_pad
    xyxy = np.zeros_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    xyxy[:, [0, 2]] -= dw
    xyxy[:, [1, 3]] -= dh
    xyxy[:, :4] /= r
    xyxy[:, 0] = np.clip(xyxy[:, 0], 0, img0_shape[1] - 1)
    xyxy[:, 1] = np.clip(xyxy[:, 1], 0, img0_shape[0] - 1)
    xyxy[:, 2] = np.clip(xyxy[:, 2], 0, img0_shape[1] - 1)
    xyxy[:, 3] = np.clip(xyxy[:, 3], 0, img0_shape[0] - 1)
    return xyxy


def nms(boxes_xyxy, scores, iou_thres=0.45, max_dets=300):
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0 and len(keep) < max_dets:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        rest = idxs[1:]
        xx1 = np.maximum(boxes_xyxy[i, 0], boxes_xyxy[rest, 0])
        yy1 = np.maximum(boxes_xyxy[i, 1], boxes_xyxy[rest, 1])
        xx2 = np.minimum(boxes_xyxy[i, 2], boxes_xyxy[rest, 2])
        yy2 = np.minimum(boxes_xyxy[i, 3], boxes_xyxy[rest, 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (boxes_xyxy[i, 2] - boxes_xyxy[i, 0]) * (boxes_xyxy[i, 3] - boxes_xyxy[i, 1])
        area_r = (boxes_xyxy[rest, 2] - boxes_xyxy[rest, 0]) * (boxes_xyxy[rest, 3] - boxes_xyxy[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-9)
        idxs = rest[iou <= iou_thres]
    return np.array(keep, dtype=np.int32)


def build_model():
    core = ov.Core()
    model = core.read_model(str(MODEL_XML), str(MODEL_BIN))
    compiled = core.compile_model(model, DEVICE)
    infer_req = compiled.create_infer_request()
    return compiled, infer_req, model.inputs[0], model.outputs[0]


def run_frame(infer_req, input_port, output_port, frame_bgr):
    img0 = frame_bgr
    img, r, (dw, dh) = letterbox(img0, IMG_SIZE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.transpose(img_rgb, (2, 0, 1))[None, ...]
    infer_req.set_tensor(input_port, ov.Tensor(array=chw))
    infer_req.infer()
    out = infer_req.get_tensor(output_port).data
    if out.ndim != 3 or out.shape[1] != 14:
        print("[WARN] Unexpected output shape:", out.shape)
        return []
    pred = out[0]
    boxes = pred[0:4, :].T
    cls_logits = pred[4:, :].T
    if USE_SOFTMAX:
        cls_scores_all = softmax(cls_logits, axis=1)
    else:
        cls_scores_all = sigmoid(cls_logits)
    class_ids = np.argmax(cls_scores_all, axis=1)
    class_scores = cls_scores_all[np.arange(cls_scores_all.shape[0]), class_ids]
    if class_scores.size > PRE_NMS_TOPK:
        topk_idx = np.argpartition(-class_scores, PRE_NMS_TOPK)[:PRE_NMS_TOPK]
        boxes = boxes[topk_idx]
        class_ids = class_ids[topk_idx]
        class_scores = class_scores[topk_idx]
    keep = class_scores >= CONF_THRES
    if not np.any(keep):
        return []
    boxes = boxes[keep]
    class_ids = class_ids[keep]
    class_scores = class_scores[keep]
    boxes_xyxy = scale_boxes((IMG_SIZE, IMG_SIZE), boxes.copy(), img0.shape, ratio_pad=(r, (dw, dh)))
    keep_idx = nms(boxes_xyxy, class_scores, IOU_THRES, MAX_DETS)
    boxes_xyxy = boxes_xyxy[keep_idx]
    class_ids = class_ids[keep_idx]
    class_scores = class_scores[keep_idx]
    dets = []
    for (x1, y1, x2, y2), cid, sc in zip(boxes_xyxy, class_ids, class_scores):
        dets.append((int(x1), int(y1), int(x2), int(y2), int(cid), float(sc)))
    return dets


def main():
    names = load_names(CLASSES_TXT)
    compiled, infer_req, input_port, output_port = build_model()

    cap = cv2.VideoCapture(0)  # your external cam index
    if not cap.isOpened():
        print("[ERROR] Could not open camera 1.")
        return

    print("[INFO] Press 'S' to save + quiz, 'Q' to quit.")
    t0 = time.time()
    frames = 0

    # Show a short on-screen hint
    subs.show("Press S to save & start quiz. Press Q to quit.", 3.0)

    queued_quiz = None  # (crop_path, cls_id)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1

        dets = run_frame(infer_req, compiled.input(0), compiled.output(0), frame)

        # draw detections
        for x1, y1, x2, y2, cid, conf in dets:
            label = f"{names[cid] if cid < len(names) else cid}: {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 2, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        # HUD
        fps = frames / max(1e-6, (time.time() - t0))
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # draw subtitles (always last before imshow)
        subs.draw(frame)
        cv2.imshow("OpenVINO YOLOv8 (raw head decode)", frame)

        # key handling
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), ord("Q")):
            break

        if k in (ord("s"), ord("S")):
            crop_path, frame_path, cls_id = save_best_detection(
                frame, dets, names, out_dir=ROOT / "screenshots"
            )
            if crop_path is not None:
                # Subtitle before leaving the live view:
                picked = names[cls_id] if 0 <= cls_id < len(names) else str(cls_id)
                subs.show(f"Saved best detection: {picked}. Launching quiz…", 2.0)

                queued_quiz = (crop_path, int(cls_id))
                # brief visual feedback
                # allow ~200 ms to actually see the message on screen
                cv2.waitKey(200)
                break  # close live view before launching quiz
            else:
                subs.show("No detections yet. Try again.", 1.8)

    cap.release()
    cv2.destroyAllWindows()

    # launch quiz if queued
    if queued_quiz is not None:
        crop_path, cls_id = queued_quiz
        run_quiz(crop_path, CLASSES_TXT, cls_id)


if __name__ == "__main__":
    main()
