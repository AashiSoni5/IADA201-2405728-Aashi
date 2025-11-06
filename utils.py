# utils.py
"""
Utility functions for drowsiness detection:
- EAR (eye aspect ratio)
- MAR (mouth aspect ratio)
- simple head-pose check (using facial landmarks)
Designed for use with MediaPipe face landmarks.
"""

import numpy as np
import math
import time
from typing import Tuple, Dict

def euclidean(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def eye_aspect_ratio(eye_points: np.ndarray) -> float:
    """
    eye_points: 6x2 array for eye landmarks (consistent order)
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    # expects shape (6,2)
    A = euclidean(eye_points[1], eye_points[5])
    B = euclidean(eye_points[2], eye_points[4])
    C = euclidean(eye_points[0], eye_points[3])
    if C == 0:
        return 0.0
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth_points: np.ndarray) -> float:
    """
    mouth_points: (8,2) or (6,2) depending on source.
    Use vertical distances / horizontal distance to estimate MAR
    """
    n = mouth_points.shape[0]
    if n >= 8:
        A = euclidean(mouth_points[1], mouth_points[7])
        B = euclidean(mouth_points[2], mouth_points[6])
        C = euclidean(mouth_points[3], mouth_points[5])
        D = euclidean(mouth_points[0], mouth_points[4])  # horizontal
    elif n >= 6:
        A = euclidean(mouth_points[1], mouth_points[5])
        B = euclidean(mouth_points[2], mouth_points[4])
        C = A
        D = euclidean(mouth_points[0], mouth_points[3])
    else:
        ys = mouth_points[:,1]
        xs = mouth_points[:,0]
        vert = ys.max() - ys.min()
        hor = xs.max() - xs.min()
        return float(vert / (hor + 1e-6))
    mar = (A + B + C) / (3.0 * (D + 1e-6))
    return mar

def head_pose_score(landmarks: np.ndarray) -> float:
    """
    Simple proxy for head pose: compute slope between nose tip and neck/base-of-chin landmarks (if available).
    Returns angle in degrees (absolute). Larger angle -> more head tilt.
    """
    try:
        left_eye = np.mean(landmarks[[33, 133]], axis=0)
        right_eye = np.mean(landmarks[[362, 263]], axis=0)
        nose = landmarks[1]
    except Exception:
        xs = landmarks[:,0]; ys = landmarks[:,1]
        left_eye = np.array([xs.min(), ys.mean()])
        right_eye = np.array([xs.max(), ys.mean()])
        nose = np.array([np.mean(xs), np.mean(ys)])
    eye_mid = (left_eye + right_eye) / 2.0
    vec = nose - eye_mid
    angle_rad = math.atan2(vec[1], vec[0] + 1e-6)
    angle_deg = abs(angle_rad * 180.0 / math.pi)
    return angle_deg

class DrowsinessDetector:
    """
    Stateful detector which accumulates counts for thresholds:
    - eye_closed_frames: counts frames where EAR < ear_thresh
    - yawning_events: counts when MAR > mar_thresh with duration
    - head_pose events: counts when angle > head_thresh
    """
    def __init__(self,
                 ear_thresh=0.22,
                 ear_consec_frames=15,
                 mar_thresh=0.6,
                 mar_consec_frames=3,
                 head_angle_thresh=20.0):
        self.ear_thresh = ear_thresh
        self.ear_consec_frames = ear_consec_frames
        self.mar_thresh = mar_thresh
        self.mar_consec_frames = mar_consec_frames
        self.head_angle_thresh = head_angle_thresh

        self.eye_counter = 0
        self.yawn_counter = 0
        self.head_counter = 0

        self.last_alert_time = 0
        self.event_log = []

    def update(self, ear, mar, head_angle, timestamp=None) -> Dict:
        """
        Call per-frame. Returns dict with state and booleans for triggers.
        """
        timestamp = timestamp or time.time()
        triggered = {"drowsy": False, "yawn": False, "distracted": False}
        # EAR: below threshold -> increment counter
        if ear < self.ear_thresh:
            self.eye_counter += 1
        else:
            if self.eye_counter >= self.ear_consec_frames:
                pass
            self.eye_counter = 0

        if self.eye_counter >= self.ear_consec_frames:
            triggered["drowsy"] = True

        # MAR: yawning detection
        if mar > self.mar_thresh:
            self.yawn_counter += 1
        else:
            if self.yawn_counter >= self.mar_consec_frames:
                triggered["yawn"] = True
            self.yawn_counter = 0

        # Head angle: simple threshold
        if head_angle > self.head_angle_thresh:
            self.head_counter += 1
        else:
            self.head_counter = 0

        if self.head_counter >= 10:
            triggered["distracted"] = True

        # If any triggered, add to event log (throttled)
        if any(triggered.values()):
            now = timestamp
            if now - self.last_alert_time > 1.0:
                e = {
                    "time": now,
                    "ear": float(ear),
                    "mar": float(mar),
                    "head_angle": float(head_angle),
                    "events": [k for k, v in triggered.items() if v]
                }
                self.event_log.append(e)
                self.last_alert_time = now

        status = "Alert"
        if triggered["drowsy"] or triggered["yawn"]:
            status = "Drowsy"
        elif triggered["distracted"]:
            status = "Distracted"

        return {"status": status, "triggered": triggered, "log_len": len(self.event_log)}
