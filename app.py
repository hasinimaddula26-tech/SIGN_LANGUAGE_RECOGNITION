"""
app.py
──────
Gradio web interface for WLASL-10 Sign Language Recognition using
ProposedResCNNBiLSTMAttn (Multi-Scale Residual Conv1D + BiLSTM + Temporal Self-Attention).

Runs locally via:
    python app.py
and is compatible out-of-the-box with Hugging Face Spaces (Gradio SDK).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.features import LandmarkExtractor
from src.models import load_model
from src.predict import Predictor

ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config.yaml"
CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / "ProposedResCNNBiLSTMAttn_best.pt"
SAMPLES_DIR = ROOT_DIR / "data" / "samples"
SAMPLES_MANIFEST = SAMPLES_DIR / "samples_labels.json"


# ─── Load configuration, checkpoint, and sample metadata ─────────────────────

def _load_resources() -> Tuple[dict, torch.nn.Module, Dict[int, str], List[dict]]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(str(CHECKPOINT_PATH), map_location=device)
    saved_labels = ckpt.get("idx_to_label") if isinstance(ckpt, dict) else None
    if saved_labels:
        idx_to_label = {int(k): str(v) for k, v in saved_labels.items()}
    else:
        idx_to_label = {i: f"class_{i}" for i in range(config.get("num_classes", 10))}

    model = load_model(str(CHECKPOINT_PATH), config, device=device)

    samples_meta: List[dict] = []
    if SAMPLES_MANIFEST.is_file():
        with open(SAMPLES_MANIFEST, "r", encoding="utf-8") as f:
            samples_meta = json.load(f)

    return config, model, idx_to_label, samples_meta


CONFIG, MODEL, IDX_TO_LABEL, SAMPLES_META = _load_resources()
SAMPLE_CHOICES = [
    f"data/samples/{item['file']}" for item in SAMPLES_META
] or [
    "data/samples/sample_1_thin.npy",
    "data/samples/sample_2_before.npy",
    "data/samples/sample_3_cool.npy",
]
SAMPLE_LOOKUP = {f"data/samples/{item['file']}": item for item in SAMPLES_META}


# ─── Visualization & Inference Helpers ───────────────────────────────────────

def plot_attention_weights(
    attn_weights: Optional[np.ndarray],
    pred_label: str,
    confidence: float,
) -> plt.Figure:
    """Create a 30-frame Temporal Attention Weight bar chart highlighting peak frames."""
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    if attn_weights is None:
        ax.text(0.5, 0.5, "Attention weights unavailable", ha="center", va="center")
        ax.axis("off")
        return fig

    frames = np.arange(len(attn_weights))
    peak_indices = set(np.argsort(attn_weights)[-5:].tolist())
    colors = ["#EF4444" if i in peak_indices else "#3B82F6" for i in frames]

    bars = ax.bar(frames, attn_weights, color=colors, edgecolor="#1E293B", linewidth=0.4)
    top_frame = int(np.argmax(attn_weights))
    ax.annotate(
        f"Peak: Frame {top_frame} ({attn_weights[top_frame]:.3f})",
        xy=(top_frame, attn_weights[top_frame]),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        fontweight="bold",
        color="#B91C1C",
    )

    ax.set_xticks(frames)
    ax.set_xticklabels([str(i) for i in frames], fontsize=8)
    ax.set_ylim(0.0, max(0.15, float(attn_weights.max()) * 1.25))
    ax.set_xlabel("Resampled Frame Index (0–29)", fontsize=10)
    ax.set_ylabel("Attention Weight", fontsize=10)
    ax.set_title(
        f"Temporal Attention Weights — Predicted: '{pred_label}' ({confidence:.1%})  "
        "[Red = Top-5 Active Gesture Frames]",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def _format_top_probs(probs: np.ndarray) -> Dict[str, float]:
    return {IDX_TO_LABEL[i]: float(probs[i]) for i in range(len(probs))}


def get_sample_ground_truth(sample_rel_path: str) -> str:
    meta = SAMPLE_LOOKUP.get(sample_rel_path)
    if not meta:
        return "Ground-truth metadata not found."
    return (
        f"**Ground-Truth Gloss:** `{meta['ground_truth_gloss']}`  |  "
        f"**Video ID:** `{meta['video_id']}`  |  "
        f"**Held-Out Signer ID:** `{meta['signer_id']}`"
    )


def predict_sample_npy(sample_rel_path: str) -> Tuple[Dict[str, float], plt.Figure, str]:
    """Run inference directly on one of the pre-extracted (30, 450) sample .npy files."""
    if not sample_rel_path:
        raise gr.Error("Please select a sample .npy file.")
    full_path = ROOT_DIR / sample_rel_path
    if not full_path.is_file():
        raise gr.Error(f"Sample file not found: {sample_rel_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = Predictor(MODEL, extractor=None, idx_to_label=IDX_TO_LABEL, device=device)
    label, conf, probs = predictor.predict_npy(str(full_path))
    fig = plot_attention_weights(predictor.last_attention, label, conf)
    return _format_top_probs(probs), fig, get_sample_ground_truth(sample_rel_path)


def predict_video_file(video_path: Optional[str]) -> Tuple[Dict[str, float], plt.Figure]:
    """Extract (30, 450) Holistic landmarks from an .mp4 / webcam recording and predict."""
    if not video_path:
        raise gr.Error("Please provide or record a video first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_len = int(CONFIG.get("sequence_length", 30))
    with LandmarkExtractor(sequence_length=seq_len, static_image_mode=False) as extractor:
        predictor = Predictor(MODEL, extractor=extractor, idx_to_label=IDX_TO_LABEL, device=device)
        label, conf, probs = predictor.predict_video(video_path)

    fig = plot_attention_weights(predictor.last_attention, label, conf)
    return _format_top_probs(probs), fig


# ─── Gradio Blocks UI ────────────────────────────────────────────────────────

with gr.Blocks(title="WLASL-10 Sign Language Recognition (ResCNN-BiLSTM-Attention)") as demo:
    gr.Markdown(
        """
        # WLASL-10 Sign Language Recognition (ResCNN-BiLSTM-Attention)
        Dynamic ASL gloss recognition across 10 words (`before`, `computer`, `cool`, `cousin`, `drink`, `go`, `help`, `tall`, `thin`, `who`)
        using **body-relative MediaPipe Holistic normalization**, **kinematic velocity features**, and **Multi-Scale Residual CNN + BiLSTM + Temporal Self-Attention**.
        """
    )

    with gr.Tabs():
        # Tab 1: Pre-Loaded Samples
        with gr.Tab("Test Pre-Loaded Samples"):
            gr.Markdown("Select a held-out `(30, 450)` test sample from `data/samples/` for instant 1-click inference.")
            sample_dropdown = gr.Dropdown(
                choices=SAMPLE_CHOICES,
                value=SAMPLE_CHOICES[0],
                label="Pre-Loaded Test Sample (.npy)",
            )
            sample_gt_md = gr.Markdown(value=get_sample_ground_truth(SAMPLE_CHOICES[0]))
            sample_btn = gr.Button("Predict Sample", variant="primary")
            with gr.Row():
                sample_label_out = gr.Label(num_top_classes=3, label="Top-3 Predicted Glosses")
                sample_attn_plot = gr.Plot(label="Temporal Attention Weights (Frames 0–29)")

            sample_dropdown.change(
                fn=get_sample_ground_truth,
                inputs=sample_dropdown,
                outputs=sample_gt_md,
            )
            sample_btn.click(
                fn=predict_sample_npy,
                inputs=sample_dropdown,
                outputs=[sample_label_out, sample_attn_plot, sample_gt_md],
            )

        # Tab 2: Upload .mp4 Video
        with gr.Tab("Upload .mp4 Video"):
            gr.Markdown("Upload an `.mp4` sign video clip to run full MediaPipe Holistic landmark extraction and inference.")
            upload_video = gr.Video(sources=["upload"], label="Upload Sign Video (.mp4)")
            upload_btn = gr.Button("Extract Landmarks & Predict", variant="primary")
            with gr.Row():
                upload_label_out = gr.Label(num_top_classes=3, label="Top-3 Predicted Glosses")
                upload_attn_plot = gr.Plot(label="Temporal Attention Weights (Frames 0–29)")

            upload_btn.click(
                fn=predict_video_file,
                inputs=upload_video,
                outputs=[upload_label_out, upload_attn_plot],
            )

        # Tab 3: Webcam Capture
        with gr.Tab("Webcam Capture"):
            gr.Markdown("Record a short sign gesture clip directly from your webcam for end-to-end landmark extraction and recognition.")
            webcam_video = gr.Video(sources=["webcam"], label="Record Webcam Sign Clip")
            webcam_btn = gr.Button("Analyze Webcam Recording", variant="primary")
            with gr.Row():
                webcam_label_out = gr.Label(num_top_classes=3, label="Top-3 Predicted Glosses")
                webcam_attn_plot = gr.Plot(label="Temporal Attention Weights (Frames 0–29)")

            webcam_btn.click(
                fn=predict_video_file,
                inputs=webcam_video,
                outputs=[webcam_label_out, webcam_attn_plot],
            )


if __name__ == "__main__":
    demo.launch()
