# One-file OpenVINO sanity check (webcam/video)
import sys, time, cv2
from pathlib import Path

# --- YOUR MODEL PATHS (exact) ---
MODEL_XML = r"D:\Thesis-Final25\Object-Detector\models\model_openvino.xml"
MODEL_BIN = r"D:\Thesis-Final25\Object-Detector\models\model_openvino.bin"
CLASSES_TXT_DEFAULT = str(Path(MODEL_XML).with_name("classes.txt"))  # optional

CONF = 0.30   # lower to 0.20 if nothing shows
SOURCE = "0"  # "0" = default webcam, or put a video path

def load_names_from_txt(p):
    p = Path(p)
    if not p.exists(): return None
    names = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s: continue
        parts = s.split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit(): names.append(parts[1])
        else: names.append(s)
    return names or None

def open_source(src):
    if str(src).isdigit(): cap = cv2.VideoCapture(int(src), cv2.CAP_DSHOW)
    else: cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError("Could not open video source. Try 0/1 or a file path.")
    return cap

def main():
    # Allow quick overrides: python sanity_openvino.py [source] [conf]
    global SOURCE, CONF
    if len(sys.argv) > 1: SOURCE = sys.argv[1]
    if len(sys.argv) > 2:
        try: CONF = float(sys.argv[2])
        except: pass

    # Check IR pair
    if not (Path(MODEL_XML).exists() and Path(MODEL_BIN).exists()):
        raise FileNotFoundError("OpenVINO IR pair not found.\n" +
                                f"  {MODEL_XML}\n  {MODEL_BIN}")

    try:
        from ultralytics import YOLO
    except Exception:
        print("\n[ERROR] Missing deps. In your venv run:")
        print("  pip install --upgrade ultralytics opencv-python openvino\n")
        raise

    print(f"[INFO] Loading: {MODEL_XML}")
    model = YOLO(MODEL_XML)  # Ultralytics uses OpenVINO backend for .xml

    # Try to get class names from model; else fall back to classes.txt
    names = None
    try:
        names = getattr(model.model, "names", None) or getattr(model, "names", None)
        if isinstance(names, dict):
            names = [names.get(i, f"id{i}") for i in range(max(names.keys()) + 1)]
    except Exception:
        pass
    if not names:
        names = load_names_from_txt(CLASSES_TXT_DEFAULT)

    cap = open_source(SOURCE)
    print("[INFO] Press 'Q' to quit.")

    prev = time.time(); fps_s = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] End of stream / cannot read frame.")
            break

        results = model(frame, conf=CONF, iou=0.45, verbose=False)
        det = results[0]

        if det.boxes is not None and len(det.boxes) > 0:
            for b in det.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                cls_id = int(b.cls[0].item()) if b.cls is not None else -1
                confv  = float(b.conf[0].item()) if b.conf is not None else 0.0
                label  = (names[cls_id] if (names and 0 <= cls_id < len(names)) else f"id{cls_id}")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,180,0), 2)
                txt = f"{label} {confv:.2f}"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0,180,0), -1)
                cv2.putText(frame, txt, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        now = time.time(); dt = now - prev; prev = now
        fps = (1.0/dt) if dt > 0 else 0.0
        fps_s = fps if fps_s == 0 else 0.9*fps_s + 0.1*fps
        cv2.putText(frame, f"FPS: {fps_s:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20,20,240), 2)

        cv2.imshow("OpenVINO Sanity Check", frame)
        if (cv2.waitKey(1) & 0xFF) in (ord('q'), ord('Q')): break

    cap.release(); cv2.destroyAllWindows(); print("[INFO] Closed.")

if __name__ == "__main__":
    main()
