# engine/detector.py
import os
import numpy as np
import cv2
import openvino as ov

# Import utils with a safe sys.path fallback
try:
    from ui.utils import letterbox, scale_boxes, nms, load_classes
except Exception:
    import sys
    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.abspath(os.path.join(HERE, ".."))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from ui.utils import letterbox, scale_boxes, nms, load_classes


def _to_numpy(x):
    try:
        return np.array(x)
    except Exception:
        return x.data if hasattr(x, "data") else x


class Detector:
    """
    OpenVINO YOLOv8 IR detector (robust):
      - Handles 1 or 2 outputs; layouts like (1,N,5+C), (1,5+C,N), (1,4+ C,N), (N,5+C), etc.
      - Tries RGB first, then BGR if no confident scores (to cover models exported expecting BGR).
      - Detects normalized coords (0..1) and scales to 640 automatically.
      - Post: conf>=0.05, class-wise NMS IoU=0.5, map back to original coords.
    """
    def __init__(self, xml_path, bin_path, classes_path, conf_thres=0.05, iou_thres=0.5):
        if not os.path.isfile(xml_path):
            raise FileNotFoundError(f"XML not found: {xml_path}")
        self.classes = load_classes(classes_path)
        self.num_classes = len(self.classes)
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.img_size = 640

        self.core = ov.Core()
        self.model = self.core.read_model(xml_path)
        self.compiled = self.core.compile_model(self.model, "CPU")
        self.input_port = self.compiled.input(0)
        self.output_ports = [self.compiled.output(i) for i in range(len(self.compiled.outputs))]
        self.infer_req = self.compiled.create_infer_request()

        # last debug info (read by UI)
        self.last_debug = {"shape": "n/a", "max_score": -1.0, "variant": "n/a", "preproc": "n/a"}

    # ---------- helpers to unify outputs ----------
    def _combine_two_outputs(self, outs):
        """Combine two-output variants: boxes (N,4) + scores (N,C)."""
        if len(outs) != 2:
            return None
        a, b = np.squeeze(outs[0]), np.squeeze(outs[1])

        def norm_boxes(x):
            if x.ndim == 3 and x.shape[0] == 1: x = x[0]
            if x.ndim == 2 and x.shape[0] == 4: x = x.T
            return x  # (N,4)

        def norm_scores(x):
            if x.ndim == 3 and x.shape[0] == 1: x = x[0]
            if x.ndim == 2 and x.shape[0] == self.num_classes: x = x.T
            return x  # (N,C)

        for boxes_raw, scores_raw in ((a, b), (b, a)):
            boxes = norm_boxes(boxes_raw)
            scores = norm_scores(scores_raw)
            if boxes.ndim == 2 and scores.ndim == 2 and boxes.shape[0] == scores.shape[0] and boxes.shape[1] == 4:
                N = boxes.shape[0]
                pred = np.zeros((N, 4 + self.num_classes), dtype=np.float32)
                pred[:, :4] = boxes
                pred[:, 4:] = scores
                return pred
        return None

    def _unify_single_output(self, out):
        """Return array shaped (N, K) with K in {4+C, 5+C}."""
        x = np.squeeze(out)
        if x.ndim == 2:
            return x
        if x.ndim == 3:
            A, B, C = x.shape
            k4 = 4 + self.num_classes
            k5 = 5 + self.num_classes
            if C in (k4, k5): return x.reshape(-1, C)
            if B in (k4, k5): return np.transpose(x, (0, 2, 1)).reshape(-1, B)
            if A in (k4, k5): return np.transpose(x, (1, 2, 0)).reshape(-1, A)
            if A == 1 and B in (k4, k5): return x[0].T
            if A == 1 and C in (k4, k5): return x[0]
        # fallback: flatten last dims
        if x.ndim > 1:
            return x.reshape(x.shape[0], -1)
        return x

    def _parse_predictions(self, arr, ratio, pad, orig_shape):
        """Turn unified array into det list."""
        if arr is None or arr.size == 0 or arr.ndim != 2:
            return [], {'ratio': ratio, 'pad': pad, 'orig_shape': orig_shape}

        K = arr.shape[1]
        k4 = 4 + self.num_classes
        k5 = 5 + self.num_classes

        # detect presence of objness
        if K >= k5:
            xywh = arr[:, 0:4]
            obj = arr[:, 4]
            cls_scores = arr[:, 5:5 + self.num_classes]
            cls_ids = np.argmax(cls_scores, axis=1)
            cls_conf = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]
            scores = obj * cls_conf
            variant = "5+C"
        elif K >= k4:
            xywh = arr[:, 0:4]
            cls_scores = arr[:, 4:4 + self.num_classes]
            cls_ids = np.argmax(cls_scores, axis=1)
            scores = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]
            variant = "4+C"
        else:
            return [], {'ratio': ratio, 'pad': pad, 'orig_shape': orig_shape}

        # Some exports output normalized coords (0..1). If widths mostly <= 1.2, scale to 640.
        if np.median(xywh[:, 2]) <= 1.2 and np.median(xywh[:, 3]) <= 1.2:
            xywh = xywh * self.img_size

        # Filter by score
        keep = scores >= self.conf_thres
        if not np.any(keep):
            self.last_debug["shape"] = f"{arr.shape}"
            self.last_debug["max_score"] = float(np.max(scores) if scores.size else -1.0)
            self.last_debug["variant"] = variant
            return [], {'ratio': ratio, 'pad': pad, 'orig_shape': orig_shape}
        xywh = xywh[keep]
        scores = scores[keep]
        cls_ids = cls_ids[keep]

        # cxcywh -> xyxy in 640 space
        cx, cy, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        dets_640 = np.stack(
            [boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3],
             scores, cls_ids.astype(np.float32)], axis=1
        )
        dets_640 = nms(dets_640, iou_thres=self.iou_thres)
        if dets_640.shape[0] == 0:
            self.last_debug["shape"] = f"{arr.shape}"
            self.last_debug["max_score"] = float(np.max(scores) if scores.size else -1.0)
            self.last_debug["variant"] = variant
            return [], {'ratio': ratio, 'pad': pad, 'orig_shape': orig_shape}

        # Map to original image coords
        boxes_orig = scale_boxes(dets_640[:, 0:4], ratio, pad, orig_shape)
        out = []
        for i in range(dets_640.shape[0]):
            x1o, y1o, x2o, y2o = boxes_orig[i].tolist()
            score = float(dets_640[i, 4])
            cid = int(dets_640[i, 5])
            out.append((x1o, y1o, x2o, y2o, score, cid))

        self.last_debug["shape"] = f"{arr.shape}"
        self.last_debug["max_score"] = float(np.max(scores))
        self.last_debug["variant"] = variant
        return out, {'ratio': ratio, 'pad': pad, 'orig_shape': orig_shape}

    # ---------- main infer ----------
    def _infer_once(self, img_float_nchw):
        result = self.infer_req.infer({self.input_port: img_float_nchw})
        outs = [_to_numpy(result[p]) for p in self.output_ports]
        if len(outs) == 1:
            unified = self._unify_single_output(outs[0])
        elif len(outs) == 2:
            combined = self._combine_two_outputs(outs)
            unified = combined if combined is not None else self._unify_single_output(
                outs[0] if outs[0].size >= outs[1].size else outs[1]
            )
        else:
            sizes = [o.size for o in outs]
            unified = self._unify_single_output(outs[int(np.argmax(sizes))])
        return unified

    def infer(self, frame_bgr):
        """
        Returns: (dets, meta) where meta contains debug info at meta['dbg'].
        """
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            return [], {'ratio': 1.0, 'pad': (0.0, 0.0), 'orig_shape': (0, 0), 'dbg': self.last_debug}

        orig_h, orig_w = frame_bgr.shape[:2]
        img_lb, r, (dw, dh) = letterbox(frame_bgr, new_shape=self.img_size)

        # Try RGB first (typical YOLOv8 export)
        rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = np.transpose(rgb, (2, 0, 1))[None, ...]
        unified = self._infer_once(inp)
        dets, meta = self._parse_predictions(unified, r, (dw, dh), (orig_h, orig_w))
        self.last_debug["preproc"] = "RGB/255"

        # If nothing came out with meaningful scores, try BGR/255 (some exports expect BGR)
        if len(dets) == 0 and (self.last_debug.get("max_score", 0.0) < 0.05):
            bgr = img_lb.astype(np.float32) / 255.0
            inp2 = np.transpose(bgr, (2, 0, 1))[None, ...]
            unified2 = self._infer_once(inp2)
            dets2, meta2 = self._parse_predictions(unified2, r, (dw, dh), (orig_h, orig_w))
            if len(dets2) > 0 or (self.last_debug.get("max_score", 0.0) < 0.05):
                dets, meta = dets2, meta2
                self.last_debug["preproc"] = "BGR/255"

        meta['dbg'] = dict(self.last_debug)  # copy
        return dets, meta
