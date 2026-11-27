"""
src/features.py
───────────────
LandmarkExtractor: MediaPipe Holistic → (T, 450) feature tensor.

Pipeline per video
──────────────────
1. Run MediaPipe Holistic on every raw frame  → 75 raw (x,y,z) landmarks/frame
2. Body-centric normalisation
   • mid_shoulder  = midpoint(pose[11], pose[12])
   • shoulder_width = ||pose[11] - pose[12]||₂  (fallback 1.0 if < 1e-6)
   • ALL 75 landmarks shifted by mid_shoulder and divided by shoulder_width
3. Kinematic velocity  Δ(x,y,z) along temporal axis  (zeros at t=0)
4. Resample to exactly 30 frames
   • > 30 frames  → uniform temporal subsampling
   • < 30 frames  → edge-padding (repeat last frame)
5. Concatenate [normalised coords | velocities] → (30, 450) float32 array
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

# ─── MediaPipe landmark indices ──────────────────────────────────────────────
_POSE_LEFT_SHOULDER  = 11
_POSE_RIGHT_SHOULDER = 12

# Counts
N_POSE      = 33
N_HAND      = 21
N_TOTAL     = N_POSE + N_HAND + N_HAND   # 75
N_COORDS    = N_TOTAL * 3                 # 225  (x, y, z)
N_FEATURES  = N_COORDS * 2               # 450  (coords + velocity)


# ─── Low-level helpers ───────────────────────────────────────────────────────

def _landmarks_to_array(
    landmark_list,
    n: int,
    world: bool = False,
) -> np.ndarray:
    """Convert a MediaPipe landmark list to an (n, 3) float32 array.

    Returns zeros if *landmark_list* is None.
    """
    out = np.zeros((n, 3), dtype=np.float32)
    if landmark_list is None:
        return out
    lms = landmark_list.landmark
    for i in range(min(n, len(lms))):
        lm = lms[i]
        out[i, 0] = lm.x
        out[i, 1] = lm.y
        out[i, 2] = lm.z
    return out


def _extract_raw_frame(result) -> np.ndarray:
    """Return (75, 3) float32 of raw (x,y,z) coordinates from one Holistic result.

    Layout:  [0:33]  = pose
             [33:54] = left hand
             [54:75] = right hand
    Undetected components → zero rows.
    """
    pose      = _landmarks_to_array(result.pose_landmarks,       N_POSE)
    left_hand = _landmarks_to_array(result.left_hand_landmarks,  N_HAND)
    right_hand= _landmarks_to_array(result.right_hand_landmarks, N_HAND)
    return np.concatenate([pose, left_hand, right_hand], axis=0)  # (75, 3)


def _normalise_frame(frame_kp: np.ndarray) -> np.ndarray:
    """Body-centric normalisation for one (75, 3) keypoint frame.

    Strategy
    --------
    • Reference origin : mid_shoulder = mean(pose[11], pose[12])
    • Scale            : shoulder_width = ||pose[11] - pose[12]||₂
                         (fallback to 1.0 if width < 1e-6, e.g. pose absent)
    • ALL detected landmarks (Pose, Left Hand, Right Hand) are shifted by
      mid_shoulder and divided by shoulder_width; completely undetected
      components (all-zero blocks) remain zero.
    """
    kp = frame_kp.copy()

    left_sh  = kp[_POSE_LEFT_SHOULDER]    # (3,)
    right_sh = kp[_POSE_RIGHT_SHOULDER]   # (3,)

    mid_shoulder   = (left_sh + right_sh) * 0.5          # (3,)
    shoulder_width = float(np.linalg.norm(left_sh - right_sh))

    if shoulder_width < 1e-6:
        shoulder_width = 1.0

    # Preserve undetected (all-zero) component blocks as exact zeros
    slices = (
        (0, N_POSE),
        (N_POSE, N_POSE + N_HAND),
        (N_POSE + N_HAND, N_TOTAL),
    )
    out = np.zeros_like(kp, dtype=np.float32)
    for start, end in slices:
        if np.any(kp[start:end]):
            out[start:end] = (kp[start:end] - mid_shoulder) / shoulder_width

    return out


def _compute_velocity(coords: np.ndarray) -> np.ndarray:
    """Frame-to-frame kinematic velocity along axis-0.

    Args:
        coords: (T, 225) normalised coordinate array.

    Returns:
        (T, 225) velocity array; first row is zeros.
    """
    vel = np.zeros_like(coords)
    vel[1:] = coords[1:] - coords[:-1]
    return vel.astype(np.float32)


def trim_active_motion(
    frames_landmarks: np.ndarray,
    margin: int = 3,
    min_active_frames: int = 8,
    threshold_ratio: float = 0.08,
) -> np.ndarray:
    """Trim leading and trailing idle frames using hand-landmark speed.

    Computes per-frame L2 norm of frame-to-frame hand coordinate differences
    across indices ``99:225`` (21 left-hand + 21 right-hand × 3 coordinates).
    Finds the contiguous window where hand speed exceeds ``8%`` of the clip's
    90th-percentile hand speed, expands by ``±margin`` frames, and falls back
    to the full clip if fewer than ``min_active_frames`` active frames exist.
    """
    T = frames_landmarks.shape[0]
    if T < min_active_frames:
        return frames_landmarks

    hands = frames_landmarks[:, 99:225]
    diffs = np.diff(hands, axis=0)
    speed = np.zeros(T, dtype=np.float32)
    speed[1:] = np.linalg.norm(diffs, axis=1)

    p90 = float(np.percentile(speed, 90))
    if p90 <= 1e-6:
        return frames_landmarks

    active_mask = speed > (threshold_ratio * p90)
    if int(active_mask.sum()) < min_active_frames:
        return frames_landmarks

    active_idx = np.where(active_mask)[0]
    start = max(0, int(active_idx[0]) - margin)
    end = min(T, int(active_idx[-1]) + margin + 1)
    return frames_landmarks[start:end]


def _resample(seq: np.ndarray, target: int = 30) -> np.ndarray:
    """Resample a (T, D) array to exactly *target* frames.

    • T > target : uniform temporal subsampling (linspace indices).
    • T < target : edge-padding (repeat last frame).
    • T == target: no-op.
    """
    T = seq.shape[0]
    if T == target:
        return seq
    if T > target:
        indices = np.round(np.linspace(0, T - 1, target)).astype(int)
        return seq[indices]
    # T < target → pad with last frame
    pad_count = target - T
    pad = np.repeat(seq[[-1]], pad_count, axis=0)
    return np.concatenate([seq, pad], axis=0)


# ─── Main class ──────────────────────────────────────────────────────────────

class LandmarkExtractor:
    """Extract (sequence_length, num_features) feature arrays from video frames.

    Parameters
    ----------
    sequence_length : int
        Target number of frames per clip (default 30).
    min_detection_confidence : float
        MediaPipe detection confidence threshold.
    min_tracking_confidence : float
        MediaPipe tracking confidence threshold.
    static_image_mode : bool
        Pass True when processing independent images (no temporal tracking).
    """

    def __init__(
        self,
        sequence_length: int = 30,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
    ) -> None:
        self.sequence_length = sequence_length
        self._mp_holistic = mp.solutions.holistic
        self._holistic = self._mp_holistic.Holistic(
            static_image_mode=static_image_mode,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        logger.info(
            "LandmarkExtractor initialised  (seq_len=%d, det=%.2f, trk=%.2f)",
            sequence_length, min_detection_confidence, min_tracking_confidence,
        )

    # ── Context-manager support ───────────────────────────────────────────────

    def __enter__(self) -> "LandmarkExtractor":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._holistic.close()

    # ── Core API ──────────────────────────────────────────────────────────────

    def process_frames(
        self,
        frames: Sequence[np.ndarray],
        trim_idle: bool = False,
    ) -> np.ndarray:
        """Convert a list of BGR frames into a (sequence_length, 450) array.

        Parameters
        ----------
        frames : list of np.ndarray
            Raw BGR uint8 frames from OpenCV (or synthetic arrays).
        trim_idle : bool
            If True, trim leading/trailing idle frames via ``trim_active_motion``
            before computing final kinematic velocity and resampling.

        Returns
        -------
        np.ndarray
            Shape (sequence_length, 450), dtype float32.
            No NaN values; undetected landmarks are zero.
        """
        if len(frames) == 0:
            logger.warning("process_frames received 0 frames; returning zeros.")
            return np.zeros(
                (self.sequence_length, N_FEATURES), dtype=np.float32
            )

        raw_coords: List[np.ndarray] = []   # each entry: (225,)

        for frame in frames:
            # Ensure uint8 RGB for MediaPipe
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self._holistic.process(rgb)

            kp75 = _extract_raw_frame(result)          # (75, 3)
            kp75 = _normalise_frame(kp75)              # body-centric
            raw_coords.append(kp75.flatten())          # (225,)

        coords = np.stack(raw_coords, axis=0)          # (T, 225)
        coords = np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
        if trim_idle:
            coords = trim_active_motion(coords, margin=3)

        vel    = _compute_velocity(coords)             # (T', 225)
        vel    = np.nan_to_num(vel,    nan=0.0, posinf=0.0, neginf=0.0)

        combined = np.concatenate([coords, vel], axis=1)  # (T', 450)
        resampled = _resample(combined, self.sequence_length)  # (30, 450)

        assert resampled.shape == (self.sequence_length, N_FEATURES), (
            f"Unexpected shape {resampled.shape}"
        )
        assert not np.isnan(resampled).any(), "NaN found in output features!"

        return resampled.astype(np.float32)

    def process_video(
        self,
        video_path: str,
        frame_start: Optional[int] = None,
        frame_end: Optional[int] = None,
    ) -> np.ndarray:
        """Extract features from a video file on disk.

        Parameters
        ----------
        video_path : str
            Path to any OpenCV-readable video file.
        frame_start : int, optional
            1-based start frame index (WLASL convention). Ignored if None or <= 0.
        frame_end : int, optional
            1-based inclusive end frame index (WLASL convention). Ignored if
            None or <= 0 (e.g. WLASL uses -1 to indicate end-of-video).

        Returns
        -------
        np.ndarray
            Shape (sequence_length, 450), dtype float32.

        Raises
        ------
        FileNotFoundError
            If the video file cannot be opened.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path!r}")

        frames: List[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        total = len(frames)
        if total > 0 and (frame_start is not None or frame_end is not None):
            start_idx = max(0, (frame_start - 1)) if (frame_start is not None and frame_start > 0) else 0
            end_idx = min(total, frame_end) if (frame_end is not None and frame_end > 0) else total
            if start_idx < end_idx:
                frames = frames[start_idx:end_idx]
            else:
                logger.warning(
                    "Invalid frame range [%s, %s] for %r (total=%d); using full video.",
                    frame_start, frame_end, video_path, total,
                )

        logger.info("Loaded %d frames from %r", len(frames), video_path)
        return self.process_frames(frames)
