# -*- coding: utf-8 -*-
"""SSI Rover - Video Detection

Runs GroundingDINO zero-shot object detection on video frames.

Usage:
    python grounding_detect.py path/to/video.mp4
    python grounding_detect.py 0                    # use webcam

Press 'q' to quit the video window.
"""

import sys
import time
import math
import copy
import json
import threading
from typing import Dict, Any, List, Optional

import torch
import cv2
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

DETECT_EVERY_N = 2
INFER_WIDTH = 480
CONFIDENCE_THRESHOLD = 0.3
CAMERA_HFOV_DEG = 60.0

KNOWN_OBJECT_WIDTHS = {
    "iphone": 0.08,
}


def compute_focal_length_px(image_width, hfov_deg):
    return (image_width / 2.0) / math.tan(math.radians(hfov_deg / 2.0))


def estimate_spatial(det, image_width, focal_length_px):
    x1, _, x2, _ = det["bbox_xyxy"]
    bbox_pixel_width = x2 - x1
    bbox_center_x = (x1 + x2) / 2.0
    image_center_x = image_width / 2.0

    angle_rad = math.atan2(bbox_center_x - image_center_x, focal_length_px)

    class_name = det["class_name"].strip().lower()
    known_width = KNOWN_OBJECT_WIDTHS.get(class_name)

    distance_m = None
    if known_width is not None and bbox_pixel_width > 0:
        distance_m = (known_width * focal_length_px) / bbox_pixel_width

    return {
        "distance_m": distance_m,
        "angle_rad": angle_rad,
        "width_m": known_width,
    }


class VideoCaptureThread:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.ret = False
        self.frame = None
        self.lock = threading.Lock()
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame
            if not ret:
                break

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def release(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()

    def get(self, prop):
        return self.cap.get(prop)

    def isOpened(self):
        return self.cap.isOpened()


def selector_to_prompt(selector):
    cls = selector.get("class_name", "")
    color = selector.get("color", "unknown")
    if color and color != "unknown":
        return f"{color} {cls}".strip()
    return cls.strip()


def bbox_bottom_center(xyxy):
    x1, _, x2, y2 = xyxy
    return float(0.5 * (x1 + x2)), float(y2)


def detect_frame(pil_img, original_size, processor, model, device, text_prompt):
    inputs = processor(images=pil_img, text=text_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        target_sizes=[original_size],
    )[0]

    boxes = results["boxes"].cpu().float().numpy().tolist()
    scores = results["scores"].cpu().float().numpy().tolist()
    labels = results["labels"]

    img_width = original_size[1]
    focal_length_px = compute_focal_length_px(img_width, CAMERA_HFOV_DEG)

    detections = []
    for i, (b, s, lab) in enumerate(zip(boxes, scores, labels)):
        if s < CONFIDENCE_THRESHOLD:
            continue
        x1, y1, x2, y2 = map(float, b)
        u, v = bbox_bottom_center([x1, y1, x2, y2])
        det = {
            "id": f"det_{i}",
            "class_name": lab,
            "confidence": float(s),
            "bbox_xyxy": [x1, y1, x2, y2],
            "pixel_uv": [u, v],
        }
        spatial = estimate_spatial(det, img_width, focal_length_px)
        det["distance_m"] = spatial["distance_m"]
        det["angle_rad"] = spatial["angle_rad"]
        det["width_m"] = spatial["width_m"]
        detections.append(det)
    return detections


def draw_detections(img_bgr, dets, current_fps):
    out = img_bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d["bbox_xyxy"])
        u, v = map(int, d["pixel_uv"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(out, (u, v), 6, (255, 0, 0), -1)
        label = f'{d["class_name"]} {d["confidence"]:.2f}'
        dist = d.get("distance_m")
        angle = d.get("angle_rad")
        if dist is not None:
            label += f' | dist={dist:.3f}m'
        if angle is not None:
            label += f' ang={math.degrees(angle):.1f}deg'
        cv2.putText(out, label, (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(out, f"FPS: {current_fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return out


def _norm(s: str) -> str:
    return s.strip().lower()


def matches_selector(det_label: str, selector: Dict[str, Any]) -> bool:
    label = _norm(det_label)
    cls = selector.get("class_name")
    color = selector.get("color")

    if cls and _norm(cls) not in label:
        return False
    if color and color != "unknown" and _norm(color) not in label:
        return False
    return True


def pick_best(dets: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not dets:
        return None
    return max(dets, key=lambda d: float(d.get("confidence", 0.0)))


def bind_detections_to_queries(
    input_queries: Dict[str, Any],
    dino_detections: List[Dict[str, Any]],
    *,
    keep_all_avoid: bool = True,
    topk_avoid: Optional[int] = None,
) -> Dict[str, Any]:
    out = {
        "detect_queries_bound": [],
        "track_queries_bound": [],
        "notes": list(input_queries.get("notes", [])),
    }

    for q in input_queries.get("detect_queries", []):
        role = q.get("role")
        selector = q.get("selector", {})
        matches = [d for d in dino_detections if matches_selector(d["class_name"], selector)]
        matches_sorted = sorted(matches, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)

        bound_entry = {
            "role": role,
            "selector": copy.deepcopy(selector),
            "matches_found": len(matches_sorted),
            "chosen": None,
            "all_matches": None,
        }

        if role in ("task_target", "task_goal"):
            chosen = pick_best(matches_sorted)
            if chosen is None:
                out["notes"].append(f"No detection found for role={role}, selector={selector}")
            bound_entry["chosen"] = chosen
        elif role == "avoid":
            if not matches_sorted:
                out["notes"].append(f"No detections found for avoid selector={selector}")
            if keep_all_avoid:
                bound_entry["all_matches"] = matches_sorted[:topk_avoid] if topk_avoid else matches_sorted
            else:
                bound_entry["chosen"] = pick_best(matches_sorted)
        else:
            bound_entry["chosen"] = pick_best(matches_sorted)

        out["detect_queries_bound"].append(bound_entry)

    for q in input_queries.get("track_queries", []):
        role = q.get("role")
        selector = q.get("selector", {})
        matches = [d for d in dino_detections if matches_selector(d["class_name"], selector)]
        chosen = pick_best(matches)

        if chosen is None:
            out["notes"].append(f"No detection found for track role={role}, selector={selector}")

        out["track_queries_bound"].append({
            "role": role,
            "selector": copy.deepcopy(selector),
            "chosen": chosen,
        })

    return out


def main():
    if len(sys.argv) < 2:
        print("Usage: python grounding_detect.py <video_path | camera_index>")
        sys.exit(1)

    video_source = sys.argv[1]
    if video_source.isdigit():
        video_source = int(video_source)

    cap = VideoCaptureThread(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{sys.argv[1]}'")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {frame_w}x{frame_h} @ {fps:.1f} FPS, {total_frames} frames")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    model_id = "IDEA-Research/grounding-dino-tiny"
    print(f"Loading model ({model_id})...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()

    vision_queries = {
        "detect_queries": [
            {"role": "task_target", "selector": {"class_name": "iphone", "color": "unknown"}},
        ],
        "track_queries": [
            {"role": "task_target", "selector": {"class_name": "iphone", "color": "unknown"}},
        ],
        "notes": [],
    }

    text_prompt = ". ".join(
        selector_to_prompt(q["selector"])
        for q in vision_queries["detect_queries"]
    ) + "."
    print(f"Text prompt: {text_prompt}")

    detect_lock = threading.Lock()
    latest_detections = []
    detect_busy = False

    def detection_worker(frame_bgr):
        nonlocal latest_detections, detect_busy
        try:
            if INFER_WIDTH and frame_w > INFER_WIDTH:
                scale = INFER_WIDTH / frame_w
                small = cv2.resize(frame_bgr, (INFER_WIDTH, int(frame_h * scale)))
                pil_img = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            else:
                pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

            dets = detect_frame(pil_img, (frame_h, frame_w), processor, model, device, text_prompt)
            with detect_lock:
                latest_detections = dets
        finally:
            detect_busy = False

    print("\nProcessing video... Press 'q' to quit.")
    frame_count = 0
    prev_time = time.time()
    current_fps = 0.0
    frame_interval = 1.0 / fps

    while True:
        loop_start = time.time()

        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_count += 1

        if not detect_busy and (frame_count % DETECT_EVERY_N == 1 or DETECT_EVERY_N == 1):
            detect_busy = True
            threading.Thread(target=detection_worker, args=(frame.copy(),), daemon=True).start()

        with detect_lock:
            detections = latest_detections

        now = time.time()
        dt = now - prev_time
        if dt > 0:
            current_fps = 1.0 / dt
        prev_time = now

        annotated = draw_detections(frame, detections, current_fps)
        cv2.imshow("GroundingDINO Video Detection", annotated)

        if frame_count % 150 == 0:
            print(f"Frame {frame_count}: {len(detections)} detections, {current_fps:.1f} FPS")
            for d in detections:
                x1, y1, x2, y2 = d["bbox_xyxy"]
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                dist = d.get("distance_m")
                angle = d.get("angle_rad")
                angle_deg = math.degrees(angle) if angle is not None else None
                dist_str = f"{dist:.3f}m" if dist is not None else "unknown"
                angle_str = f"{angle_deg:.1f}deg" if angle_deg is not None else "unknown"
                print(f"  {d['class_name']} conf={d['confidence']:.2f} "
                      f"bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] "
                      f"size={bbox_w:.0f}x{bbox_h:.0f}px "
                      f"center=({d['pixel_uv'][0]:.0f},{d['pixel_uv'][1]:.0f}) "
                      f"dist={dist_str} angle={angle_str}")

        elapsed = time.time() - loop_start
        wait_ms = max(1, int((frame_interval - elapsed) * 1000))
        if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
            print("Quit by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if detections:
        bound = bind_detections_to_queries(vision_queries, detections, keep_all_avoid=True)
        print("\nFinal frame bound results:")
        print(json.dumps(bound, indent=2))


if __name__ == "__main__":
    main()
