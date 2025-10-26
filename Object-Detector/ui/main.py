# ui/main.py
import os
import sys
import cv2
import numpy as np
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor

from PyQt6.QtCore import Qt, QTimer, QPoint, QSize
from PyQt6.QtGui import QImage, QPixmap, QAction
from PyQt6.QtWidgets import (
    QWidget, QLabel, QMainWindow, QApplication, QMessageBox, QStatusBar, QFileDialog
)

# Robust relative imports
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from engine.detector import Detector
from ui.utils import draw_boxes, load_classes

# Try to import the quiz dialog (will be provided in STEP 3)
QUIZ_AVAILABLE = True
try:
    from ui.quiz_dialog import QuizDialog
except Exception:
    QUIZ_AVAILABLE = False


class VideoWidget(QLabel):
    """
    QLabel that shows video and handles mouse clicks to detect box selection.
    """
    def __init__(self, click_callback):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: black;")
        self._click_callback = click_callback
        self.setMinimumSize(640, 480)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
            self._click_callback(pos)


class MainWindow(QMainWindow):
    def __init__(self,
                 xml_path="models/model_openvino.xml",
                 bin_path="models/model_openvino.bin",
                 classes_path="models/classes.txt",
                 cam_index_primary=0,
                 cam_index_fallback=1):
        super().__init__()
        self.setWindowTitle("Thesis25 — Object Detector (OpenVINO + PyQt6)")
        self.resize(960, 740)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        # Central video widget
        self.video_label = VideoWidget(self.on_click)
        self.setCentralWidget(self.video_label)

        # Menu: File -> Open classes.txt (optional)
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        open_classes_act = QAction("Load classes.txt…", self)
        open_classes_act.triggered.connect(self._choose_classes)
        file_menu.addAction(open_classes_act)

        # Load classes
        self.classes = load_classes(classes_path)

        # Detector
        self.detector = Detector(xml_path, bin_path, classes_path, conf_thres=0.25, iou_thres=0.5)

        # Camera setup with fallback
        self.cap_index = None
        self.cap = None
        if self._open_camera(cam_index_primary):
            self.cap_index = cam_index_primary
            self.status.showMessage(f"Webcam opened: index {cam_index_primary}")
        elif self._open_camera(cam_index_fallback):
            self.cap_index = cam_index_fallback
            self.status.showMessage(f"Primary webcam failed; using fallback index {cam_index_fallback}")
        else:
            QMessageBox.critical(self, "Camera Error", "Unable to open webcam (tried 0 and 1).")
            self.status.showMessage("No camera available.")
            return

        # Threaded inference: single worker, drop frames if busy
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending_future = None
        self.last_frame_bgr = None
        self.last_dets: List[Tuple[float, float, float, float, float, int]] = []

        # Timer ~30 FPS for grabbing frames & dispatching inference
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)  # ~30fps

    def _choose_classes(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select classes.txt", ROOT, "Text Files (*.txt)")
        if path:
            try:
                self.classes = load_classes(path)
                self.status.showMessage(f"Loaded classes from: {path}")
            except Exception as e:
                QMessageBox.warning(self, "Load Classes Error", str(e))

    def _open_camera(self, index: int):
        # Try default backend first (works for most cameras including Orbbec)
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            # Fallback to OBSENSOR backend for Orbbec cameras
            cap = cv2.VideoCapture(index, cv2.CAP_OBSENSOR)
        if not cap.isOpened():
            return False
        # Try to set a decent resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap = cap
        return True

    def closeEvent(self, event):
        try:
            if self.cap:
                self.cap.release()
            if self.executor:
                self.executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        event.accept()

    def _tick(self):
        if self.cap is None:
            return
        ok, frame_bgr = self.cap.read()
        if not ok or frame_bgr is None:
            self.status.showMessage("Failed to read from camera.")
            return

        self.last_frame_bgr = frame_bgr

        # Dispatch inference if no job pending
        if self.pending_future is None or self.pending_future.done():
            self.pending_future = self.executor.submit(self.detector.infer, frame_bgr.copy())

        # If inference result ready, consume it
        dbg_text = ""
        if self.pending_future and self.pending_future.done():
            try:
                dets, meta = self.pending_future.result()
                self.last_dets = dets or []
                dbg = meta.get('dbg', {})
                dbg_text = f"out:{dbg.get('shape','?')}  max:{dbg.get('max_score',-1):.3f}  var:{dbg.get('variant','?')}  pre:{dbg.get('preproc','?')}"
            except Exception as e:
                self.last_dets = []
                dbg_text = f"Inference error: {e}"

        # Draw boxes
        frame_drawn = draw_boxes(frame_bgr, self.last_dets, self.classes)
        self._set_image(frame_drawn)

        # Status
        if dbg_text:
            self.status.showMessage(dbg_text)


    def _set_image(self, frame_bgr):
        # Convert BGR -> RGB -> QImage -> QLabel
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        # Scale to label while keeping aspect
        pix = QPixmap.fromImage(qimg)
        target = self.video_label.size()
        self.video_label.setPixmap(pix.scaled(target, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def on_click(self, pos: QPoint):
        """
        Called when user clicks on the video label.
        Determine if click falls into any detection box; if so, open quiz dialog.
        """
        if self.last_frame_bgr is None or not self.last_dets:
            return

        label_pix = self.video_label.pixmap()
        if label_pix is None:
            return

        # Compute scale/offset applied when scaling pixmap to label
        label_size: QSize = self.video_label.size()
        pix_w, pix_h = label_pix.width(), label_pix.height()
        lbl_w, lbl_h = label_size.width(), label_size.height()

        scale = min(lbl_w / pix_w, lbl_h / pix_h)
        disp_w = int(pix_w * scale)
        disp_h = int(pix_h * scale)
        off_x = (lbl_w - disp_w) // 2
        off_y = (lbl_h - disp_h) // 2

        # Click within displayed image region?
        x = pos.x() - off_x
        y = pos.y() - off_y
        if x < 0 or y < 0 or x >= disp_w or y >= disp_h:
            return

        # Map to original frame coords
        frame_h, frame_w = self.last_frame_bgr.shape[:2]
        img_x = x * (frame_w / disp_w)
        img_y = y * (frame_h / disp_h)

        # Find first detection containing the point
        hit = None
        for (x1, y1, x2, y2, score, cid) in self.last_dets:
            if img_x >= x1 and img_x <= x2 and img_y >= y1 and img_y <= y2:
                hit = (int(x1), int(y1), int(x2), int(y2), float(score), int(cid))
                break

        if not hit:
            return

        # Freeze current frame and crop ROI
        x1, y1, x2, y2, score, cid = hit
        frame = self.last_frame_bgr.copy()
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(frame_w - 1, x2), min(frame_h - 1, y2)
        crop = frame[y1c:y2c, x1c:x2c].copy()
        correct_name = self.classes[cid] if 0 <= cid < len(self.classes) else f"id{cid}"

        if QUIZ_AVAILABLE:
            try:
                dlg = QuizDialog(self, crop, correct_name, self.classes)
                dlg.exec()
            except Exception as e:
                QMessageBox.warning(self, "Quiz Error", f"Quiz failed to open:\n{e}")
        else:
            QMessageBox.information(
                self,
                "Quiz not ready",
                "Quiz dialog will be added in STEP 3.\nYour click detection is working!"
            )
