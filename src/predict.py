"""
src/predict.py
──────────────
Real-time webcam inference and single-video prediction.

Usage – webcam
──────────────
    python -m src.predict --config config.yaml \\
                          --checkpoint models/best_model.pt \\
                          --mode webcam

Usage – single video
────────────────────
    python -m src.predict --config config.yaml \\
                          --checkpoint models/best_model.pt \\
                          --mode video --video path/to/clip.mp4

Usage (Python API)
──────────────────
    from src.predict import Predictor
    predictor = Predictor(model, extractor, idx_to_label, device)
    label, confidence, all_probs = predictor.predict_video("clip.mp4")
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml

from src.features import LandmarkExtractor
from src.models import load_model

logger = logging.getLogger(__name__)


# ─── Predictor ───────────────────────────────────────────────────────────────

class Predictor:
    """Single-inference wrapper supporting all 4 architectures."""

    def __init__(
        self,
        model: nn.Module,
        extractor: Optional[LandmarkExtractor],
        idx_to_label: Dict[int, str],
        device: str = "cpu",
    ) -> None:
        self.model        = model.to(device)
        self.model.eval()
        self.extractor    = extractor
        self.idx_to_label = idx_to_label
        self.device       = torch.device(device)
        self.last_attention: Optional[np.ndarray] = None

    # ── Core inference ────────────────────────────────────────────────────────

    def _infer_features(self, feat: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """Run model on a (30, 450) numpy array.

        Returns (label, confidence, all_probs). For attention-equipped models,
        also stores ``self.last_attention`` of shape ``(30,)``.
        """
        x = torch.from_numpy(feat).unsqueeze(0).to(self.device)  # (1, 30, 450)
        with torch.no_grad():
            out = self.model(x)
            if isinstance(out, tuple):
                logits, attn = out
                self.last_attention = attn.squeeze(0).cpu().numpy()
            else:
                logits = out
                self.last_attention = None
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx   = int(probs.argmax())
        label      = self.idx_to_label.get(pred_idx, str(pred_idx))
        confidence = float(probs[pred_idx])
        return label, confidence, probs

    def predict_frames(
        self, frames: List[np.ndarray]
    ) -> Tuple[str, float, np.ndarray]:
        """Predict class from a list of BGR frames."""
        if self.extractor is None:
            raise RuntimeError("LandmarkExtractor is required for frame/video prediction.")
        feat = self.extractor.process_frames(frames)
        return self._infer_features(feat)

    def predict_video(self, video_path: str) -> Tuple[str, float, np.ndarray]:
        """Predict class from a video file."""
        if self.extractor is None:
            raise RuntimeError("LandmarkExtractor is required for video prediction.")
        feat = self.extractor.process_video(video_path)
        return self._infer_features(feat)

    def predict_npy(self, npy_path: str) -> Tuple[str, float, np.ndarray]:
        """Predict class directly from a pre-extracted (30, 450) .npy landmark file."""
        feat = np.load(npy_path).astype(np.float32)
        return self._infer_features(feat)


# ─── Webcam inference ────────────────────────────────────────────────────────

def run_webcam(
    predictor: Predictor,
    sequence_length: int = 30,
    camera_id: int = 0,
    dry_run: bool = False,
    dry_run_frames: Optional[List[np.ndarray]] = None,
) -> None:
    """Real-time webcam inference with OpenCV overlay. Press 'q' to quit."""
    cap = None if dry_run else cv2.VideoCapture(camera_id)
    if not dry_run and not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")

    buffer: deque = deque(maxlen=sequence_length)
    label, confidence = "Collecting...", 0.0
    last_pred_time = 0.0

    logger.info("Webcam inference started%s. Press 'q' to quit.", " (dry-run)" if dry_run else "")
    frame_iter = iter(dry_run_frames or [])
    step = 0
    try:
        while True:
            if dry_run:
                try:
                    frame = next(frame_iter)
                    ret = True
                except StopIteration:
                    break
            else:
                ret, frame = cap.read()  # type: ignore[union-attr]
            if not ret:
                break

            step += 1
            buffer.append(frame.copy())

            now = time.time()
            if len(buffer) == sequence_length and (dry_run or now - last_pred_time > 1.0):
                label, confidence, _ = predictor.predict_frames(list(buffer))
                last_pred_time = now

            h, w = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            cv2.putText(
                frame,
                f"{label}  ({confidence:.1%})",
                (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 80), 2, cv2.LINE_AA,
            )

            if not dry_run:
                cv2.imshow("Sign Language Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if cap is not None:
            cap.release()
        if not dry_run:
            cv2.destroyAllWindows()
    if dry_run:
        logger.info("Webcam dry-run completed cleanly (%d frames processed; last=%s %.1f%%).", step, label, confidence * 100)


# ─── CLI entry point ─────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sign Language Predictor")
    p.add_argument("--config",     default="config.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--labels",     default=None, help="Optional JSON: int→label mapping (falls back to checkpoint)")
    p.add_argument("--model",      default=None, help="Optional model name override")
    p.add_argument("--mode",       choices=["webcam", "video", "npy"], default="webcam")
    p.add_argument("--video",      default=None,  help="Video or .npy path (--mode video/npy)")
    p.add_argument("--input",      default=None,  help="Input .npy or video file path")
    p.add_argument("--camera",     type=int, default=0)
    p.add_argument("--dry-run",    action="store_true", help="Verify webcam pipeline without opening GUI/camera")
    p.add_argument("--device",     default="auto")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    args = _parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    idx_to_label: Dict[int, str]
    if args.labels:
        with open(args.labels, encoding="utf-8") as f:
            raw = json.load(f)
        idx_to_label = {int(k): v for k, v in raw.items()}
    else:
        ckpt = torch.load(args.checkpoint, map_location=device)
        saved = ckpt.get("idx_to_label") if isinstance(ckpt, dict) else None
        if saved:
            idx_to_label = {int(k): v for k, v in saved.items()}
        else:
            idx_to_label = {i: f"class_{i}" for i in range(config.get("num_classes", 10))}

    model = load_model(args.checkpoint, config, device=device, model_name=args.model)
    target_path = args.input or args.video
    is_npy = args.mode == "npy" or (target_path is not None and target_path.lower().endswith(".npy"))

    extractor = (
        None
        if is_npy
        else LandmarkExtractor(
            sequence_length=config.get("sequence_length", 30),
            static_image_mode=False,
        )
    )

    predictor = Predictor(model, extractor, idx_to_label, device=device)

    if args.mode == "webcam" and not target_path:
        seq_len = config.get("sequence_length", 30)
        dry_frames = (
            [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(seq_len + 2)]
            if args.dry_run
            else None
        )
        run_webcam(
            predictor,
            sequence_length=seq_len,
            camera_id=args.camera,
            dry_run=args.dry_run,
            dry_run_frames=dry_frames,
        )
    else:
        if not target_path:
            raise ValueError("--input (or --video) is required when --mode is 'video' or 'npy'")
        if is_npy:
            label, conf, probs = predictor.predict_npy(target_path)
        else:
            label, conf, probs = predictor.predict_video(target_path)
        logger.info("Input: %s | Prediction: %s  (confidence=%.2f%%)", target_path, label, conf * 100)
        if predictor.last_attention is not None:
            top_frames = np.argsort(predictor.last_attention)[::-1][:5].tolist()
            logger.info("Top-5 attended frame indices: %s", top_frames)
        for i, p in enumerate(probs):
            logger.info("  %s : %.4f", idx_to_label.get(i, str(i)), p)


if __name__ == "__main__":
    main()
