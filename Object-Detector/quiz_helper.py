# quiz_helper.py
import cv2, time, random
from pathlib import Path

# ---------- Subtitle helper (built-in) ----------
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
        cv2.putText(frame, self.text, (12, h - banner_h + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

subs = SubtitleBar()
# ------------------------------------------------

# Optional offline TTS (safe to ignore if not installed)
try:
    import pyttsx3
    _TTS = pyttsx3.init()
    _TTS.setProperty("rate", 180)
except Exception:
    _TTS = None

def _speak(text: str, subtitle_secs: float = 2.5):
    # Always show subtitles that match TTS
    subs.show(text, subtitle_secs)
    if _TTS:
        try:
            _TTS.stop(); _TTS.say(text); _TTS.runAndWait()
        except Exception:
            pass

def _load_names(p: Path):
    try:
        return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except Exception:
        return [f"cls{i}" for i in range(10)]

def _fit(img, max_w=1100, max_h=800):
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    return cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA) if s < 1.0 else img

def run_quiz(image_path: str, classes_txt: Path, correct_cls_id: int):
    """
    Opens a quiz window on the saved image.
    - Options are shown WITHOUT numbers, just names.
    - Click an option (or press 1..4) to answer.
    - Click 🔊 Speak to hear the sentence: "This is <name>."
    - Subtitles always mirror the spoken text.
    - On wrong answer, it says: "No, it is not <wrong>. It is <correct>."
    - On correct answer, it says: "Yes, this is <correct>."
    """
    names = _load_names(classes_txt)
    correct_cls_id = int(correct_cls_id)
    correct_name = names[correct_cls_id] if 0 <= correct_cls_id < len(names) else str(correct_cls_id)

    img = cv2.imread(image_path)
    if img is None:
        print("[ERR] cannot read:", image_path)
        return

    base = _fit(img)
    h, w = base.shape[:2]

    # Build 4 options: correct + 3 random distractors
    pool = [i for i in range(len(names)) if i != correct_cls_id]
    random.shuffle(pool)
    options = [correct_cls_id] + pool[:3]
    random.shuffle(options)
    option_texts = [names[i] for i in options]

    # Layout
    speak_rect = (w-180, 20, w-20, 70)
    opt_x = 20
    opt_y0 = 85
    line_gap = 38
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.95
    thickness = 2

    # Precompute clickable rectangles for options (based on text size)
    option_boxes = []
    for idx, text in enumerate(option_texts):
        y_base = opt_y0 + idx * line_gap
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x1, y1 = opt_x, y_base - th
        x2, y2 = opt_x + tw, y_base
        option_boxes.append((x1, y1, x2, y2))

    # State
    speaking_until = 0.0
    selected_index = None  # 0..3 when user chooses
    feedback_text = None

    def render(speaking=False, feedback=None):
        c = base.copy()
        cv2.putText(c, "What object is this?", (20, 40),
                    font, 1.0, (50, 220, 255), 3, cv2.LINE_AA)

        # Draw options (NO numbers, just names)
        for i, text in enumerate(option_texts):
            y_base = opt_y0 + i * line_gap
            cv2.putText(c, text, (opt_x, y_base),
                        font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

        # Speak button
        (x1,y1,x2,y2) = speak_rect
        color = (0,200,0) if speaking else (255,255,255)
        cv2.rectangle(c, (x1,y1), (x2,y2), (40,40,40), -1)
        cv2.rectangle(c, (x1,y1), (x2,y2), color, 2)
        cv2.putText(c, "Speak", (x1+20, y1+32), font, 0.8, color, 2, cv2.LINE_AA)

        # Feedback text
        if feedback:
            col = (0,200,0) if feedback.startswith("Correct") or feedback.startswith("Yes") else (0,0,255)
            cv2.putText(c, feedback, (20, opt_y0 + 4*line_gap + 10),
                        font, 1.1, col, 3, cv2.LINE_AA)

        # Draw subtitles last so they sit on top
        subs.draw(c)
        return c

    def on_mouse(event, x, y, flags, param):
        nonlocal speaking_until, selected_index
        if event == cv2.EVENT_LBUTTONDOWN:
            # Speak button
            x1,y1,x2,y2 = speak_rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                # Say "This is <name>." and show same as subtitle
                phrase = f"This is {correct_name}."
                _speak(phrase)
                speaking_until = time.time() + 0.8
                return
            # Option clicks
            for i, (ox1, oy1, ox2, oy2) in enumerate(option_boxes):
                if ox1 <= x <= ox2 and oy1 <= y <= oy2:
                    selected_index = i
                    return

    cv2.namedWindow("Quiz", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Quiz", on_mouse)

    while True:
        now = time.time()

        # Keyboard selection (still supported)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q')):  # ESC or q to quit
            break
        if k in (ord('1'), ord('2'), ord('3'), ord('4')):
            selected_index = int(chr(k)) - 1

        # If user made a selection (by click or key)
        if selected_index is not None and 0 <= selected_index < 4:
            chosen_cls = options[selected_index]
            chosen_name = names[chosen_cls] if 0 <= chosen_cls < len(names) else str(chosen_cls)
            if chosen_cls == correct_cls_id:
                feedback_text = f"Yes, this is {correct_name}."
                _speak(feedback_text)
            else:
                feedback_text = f"No, it is not {chosen_name}. It is {correct_name}."
                _speak(feedback_text)
            # Show feedback for a moment, then exit
            cv2.imshow("Quiz", render(speaking=False, feedback=feedback_text))
            cv2.waitKey(1500)
            break

        # Normal redraw
        cv2.imshow("Quiz", render(speaking=(now < speaking_until), feedback=feedback_text))

    cv2.destroyWindow("Quiz")
