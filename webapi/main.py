# webapi/main.py — detect + quiz options for web UI
from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import time, random, re
import cv2, numpy as np, openvino as ov

# ---------- Paths relative to repo root ----------
ROOT = Path(__file__).resolve().parent.parent / "Object-Detector"
MODEL_XML = ROOT / "models" / "model_openvino.xml"
MODEL_BIN = ROOT / "models" / "model_openvino.bin"
CLASSES_TXT = ROOT / "models" / "classes.txt"

# ---------- Config ----------
DEVICE = "CPU"
IMG_SIZE = 640
CONF_THRES = 0.65          # slightly lower to be more permissive
IOU_THRES = 0.45
MAX_DETS = 100
PRE_NMS_TOPK = 300
USE_SOFTMAX = False

# ---------- Utils ----------
def _clean_label(s: str) -> str:
    """
    Remove any leading numeric index and common separators:
      '0 brush' -> 'brush'
      '1) pen'  -> 'pen'
      '2. cup'  -> 'cup'
      '03 - book' -> 'book'
    Keeps inner spaces; trims ends.
    """
    s = s.strip()
    s = re.sub(r'^\s*\d+\s*[:\-\.\)\]]*\s*', '', s)
    return s.strip()

def load_names(p: Path):
    if not p.exists():
        print(f"[WARN] classes.txt not found at {p}")
        # Fallback generic names
        return [f"cls{i}" for i in range(10)]
    raw = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    cleaned = [_clean_label(ln) for ln in raw]
    # de-dup while preserving order
    seen = set()
    out = []
    for n in cleaned:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out

def sigmoid(x): return 1.0/(1.0+np.exp(-x))
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x); return e/(np.sum(e, axis=axis, keepdims=True)+1e-9)

def letterbox(im, new_shape=640, color=(114,114,114)):
    h, w = im.shape[:2]
    if isinstance(new_shape, int): new_shape = (new_shape, new_shape)
    r = min(new_shape[0]/h, new_shape[1]/w)
    new_unpad = (int(round(w*r)), int(round(h*r)))
    dw = new_shape[1]-new_unpad[0]; dh = new_shape[0]-new_unpad[1]
    dw//=2; dh//=2
    if (w,h)!=new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(im, dh, new_shape[0]-new_unpad[1]-dh, dw,
                            new_shape[1]-new_unpad[0]-dw, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

def scale_boxes(boxes, img0_shape, ratio_pad):
    r,(dw,dh)=ratio_pad
    xyxy=np.zeros_like(boxes)
    xyxy[:,0]=boxes[:,0]-boxes[:,2]/2
    xyxy[:,1]=boxes[:,1]-boxes[:,3]/2
    xyxy[:,2]=boxes[:,0]+boxes[:,2]/2
    xyxy[:,3]=boxes[:,1]+boxes[:,3]/2
    xyxy[:,[0,2]]-=dw; xyxy[:,[1,3]]-=dh; xyxy[:,:4]/=r
    H, W = img0_shape[:2]
    xyxy[:,0]=np.clip(xyxy[:,0],0,W-1)
    xyxy[:,1]=np.clip(xyxy[:,1],0,H-1)
    xyxy[:,2]=np.clip(xyxy[:,2],0,W-1)
    xyxy[:,3]=np.clip(xyxy[:,3],0,H-1)
    return xyxy

def nms(boxes_xyxy, scores, iou_thres=0.45, max_dets=300):
    idxs = scores.argsort()[::-1]; keep=[]
    while idxs.size>0 and len(keep)<max_dets:
        i = idxs[0]; keep.append(i)
        if idxs.size==1: break
        rest = idxs[1:]
        xx1=np.maximum(boxes_xyxy[i,0], boxes_xyxy[rest,0])
        yy1=np.maximum(boxes_xyxy[i,1], boxes_xyxy[rest,1])
        xx2=np.minimum(boxes_xyxy[i,2], boxes_xyxy[rest,2])
        yy2=np.minimum(boxes_xyxy[i,3], boxes_xyxy[rest,3])
        w=np.maximum(0.0, xx2-xx1); h=np.maximum(0.0, yy2-yy1)
        inter=w*h
        area_i=(boxes_xyxy[i,2]-boxes_xyxy[i,0])*(boxes_xyxy[i,3]-boxes_xyxy[i,1])
        area_r=(boxes_xyxy[rest,2]-boxes_xyxy[rest,0])*(boxes_xyxy[rest,3]-boxes_xyxy[rest,1])
        iou=inter/(area_i+area_r-inter+1e-9)
        idxs=rest[iou<=iou_thres]
    return np.array(keep, dtype=np.int32)

# ---------- Load model once ----------
core = ov.Core()
print(f"[INFO] Loading IR:\n  {MODEL_XML}\n  {MODEL_BIN}")
model = core.read_model(str(MODEL_XML), str(MODEL_BIN))
compiled = core.compile_model(model, DEVICE)
infer_req = compiled.create_infer_request()
input_port = model.inputs[0]; output_port = model.outputs[0]
names = load_names(CLASSES_TXT)
print(f"[INFO] Classes: {len(names)}")

def infer_bgr(frame_bgr):
    img0 = frame_bgr
    img, r, (dw, dh) = letterbox(img0, IMG_SIZE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.transpose(img_rgb, (2,0,1))[None, ...]
    infer_req.set_tensor(input_port, ov.Tensor(array=chw))
    infer_req.infer()
    out = infer_req.get_tensor(output_port).data  # common shapes: [1,14,N] or [1,N,14]

    # Try to interpret both layouts
    boxes = None; cls_logits = None
    if out.ndim==3:
        p = out[0]
        # A) [(4+C), N]
        if p.ndim==2 and p.shape[0] >= 5:
            C = p.shape[0]-4
            boxes = p[0:4,:].T
            cls_logits = p[4:4+C,:].T
        # B) [N, (4+C)]
        if (boxes is None or cls_logits is None) and p.ndim==2 and p.shape[1] >= 5:
            C = p.shape[1]-4
            boxes = p[:,0:4]
            cls_logits = p[:,4:4+C]

    if boxes is None or cls_logits is None:
        print("[WARN] Could not interpret output tensor, shape:", out.shape)
        return []

    cls_scores_all = softmax(cls_logits, axis=1) if USE_SOFTMAX else sigmoid(cls_logits)
    class_ids = np.argmax(cls_scores_all, axis=1)
    class_scores = cls_scores_all[np.arange(cls_scores_all.shape[0]), class_ids]

    if class_scores.size > PRE_NMS_TOPK:
        topk = np.argpartition(-class_scores, PRE_NMS_TOPK)[:PRE_NMS_TOPK]
        boxes, class_ids, class_scores = boxes[topk], class_ids[topk], class_scores[topk]

    keep = class_scores >= CONF_THRES
    if not np.any(keep):
        return []

    boxes = boxes[keep]; class_ids = class_ids[keep]; class_scores = class_scores[keep]
    boxes_xyxy = scale_boxes(boxes.copy(), frame_bgr.shape, (r,(dw,dh)))
    keep_idx = nms(boxes_xyxy, class_scores, IOU_THRES, MAX_DETS)
    boxes_xyxy, class_ids, class_scores = boxes_xyxy[keep_idx], class_ids[keep_idx], class_scores[keep_idx]

    dets = []
    for (x1,y1,x2,y2), cid, sc in zip(boxes_xyxy, class_ids, class_scores):
        label = names[cid] if 0 <= cid < len(names) else str(int(cid))
        dets.append({
            "cls": label,
            "conf": float(sc),
            "xyxy": [int(x1), int(y1), int(x2), int(y2)]
        })
    return dets

# ---------- API ----------
app = FastAPI(title="Thesis Detector API")

# Allow CRA on 3000/3001
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000","http://127.0.0.1:3000",
        "http://localhost:3001","http://127.0.0.1:3001",
    ],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
def health(): return {"ok": True, "classes_count": len(names)}

@app.get("/classes")
def get_classes(): return {"classes": names}

@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    t0 = time.time()
    arr = np.frombuffer(await image.read(), np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None: return {"detections": [], "best": None, "latency_ms": 0}
    dets = infer_bgr(frame)
    best = None
    if dets:
        top = max(dets, key=lambda d: d["conf"])
        best = {"cls": top["cls"], "conf": top["conf"]}
    return {"detections": dets, "best": best, "latency_ms": int((time.time()-t0)*1000)}

@app.get("/quiz/options")
def quiz_options(correct: str = Query(..., description="Correct class name"), k: int = 4):
    """Return k options (1 correct + k-1 distractors) shuffled."""
    uniq = []
    seen = set()
    for n in names:
        if n not in seen:
            uniq.append(n); seen.add(n)
    # case-insensitive match for safety
    corr = correct
    pool = [n for n in uniq if n.lower() != corr.lower()]
    random.shuffle(pool)
    picked = [corr] + pool[:max(0, k-1)]
    random.shuffle(picked)
    return {"options": picked, "correct_index": picked.index(corr)}
