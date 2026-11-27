"""
src/evaluate.py
───────────────
Evaluation, 4-model benchmarking, CPU warm-up latency measurement, hardware
logging, CSV export, and 300-DPI presentation chart generation.

Usage (command line)
────────────────────
    # Single model evaluation
    python -m src.evaluate --config config.yaml --checkpoint models/best_model.pt --data data/wlasl10_landmarks.npz

    # Benchmark all 4 models from checkpoints/ and export tables + 4 plots to results/
    python -m src.evaluate --config config.yaml --data data/wlasl10_landmarks.npz --benchmark-all
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)
from torch.utils.data import DataLoader

from src.dataset import build_dataloaders
from src.models import load_model

logger = logging.getLogger(__name__)

ALL_MODEL_NAMES: List[str] = [
    "CNNBaseline",
    "LSTMBaseline",
    "CNNLSTMStandard",
    "ProposedResCNNBiLSTMAttn",
]

COMPARISON_COLUMNS: List[str] = [
    "Model Name",
    "Top-1 Accuracy",
    "Top-3 Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1-Score",
    "Weighted F1-Score",
    "Parameter Count",
    "Checkpoint Size (KB)",
    "CPU Latency Mean (ms)",
    "CPU Latency Std (ms)",
]


# ─── Hardware & Latency Helpers ──────────────────────────────────────────────

def collect_hardware_specs() -> Dict[str, object]:
    """Collect OS, CPU, Python, and PyTorch version metadata."""
    return {
        "os": platform.platform(),
        "os_system": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "cpu": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_num_threads": torch.get_num_threads(),
    }


def measure_cpu_latency(
    model: nn.Module,
    sequence_length: int = 30,
    num_features: int = 450,
    warmup_runs: int = 20,
    timed_runs: int = 100,
) -> Tuple[float, float]:
    """Measure single-sample CPU inference latency in milliseconds.

    Runs *warmup_runs* forward passes first, then times *timed_runs* passes
    with ``batch_size=1`` in ``eval()`` mode under ``torch.no_grad()``.
    """
    cpu_model = model.to("cpu")
    cpu_model.eval()
    dummy = torch.randn(1, sequence_length, num_features, device="cpu")

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = cpu_model(dummy)

        timings_ms: List[float] = []
        for _ in range(timed_runs):
            t0 = time.perf_counter()
            _ = cpu_model(dummy)
            timings_ms.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(timings_ms)), float(np.std(timings_ms))


# ─── Evaluator ───────────────────────────────────────────────────────────────

class Evaluator:
    """Run inference on a DataLoader and compute classification metrics."""

    def __init__(
        self,
        model: nn.Module,
        loader: DataLoader,
        idx_to_label: Dict[int, str],
        device: str | torch.device = "cpu",
    ) -> None:
        self.model        = model
        self.loader       = loader
        self.idx_to_label = idx_to_label
        self.device       = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.last_attention_weights: Optional[np.ndarray] = None

    # ── Inference ─────────────────────────────────────────────────────────────

    def _collect_predictions(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        all_labels:  List[int]        = []
        all_preds:   List[int]        = []
        all_probs:   List[np.ndarray] = []
        all_attn:    List[np.ndarray] = []

        with torch.no_grad():
            for x, y in self.loader:
                x = x.to(self.device)
                out = self.model(x)
                if isinstance(out, tuple):
                    logits, attn = out
                    all_attn.append(attn.cpu().numpy())
                else:
                    logits = out
                probs  = torch.softmax(logits, dim=1).cpu().numpy()
                preds  = logits.argmax(dim=1).cpu().numpy()

                all_labels.extend(y.numpy().tolist())
                all_preds.extend(preds.tolist())
                all_probs.append(probs)

        attn_arr = np.vstack(all_attn) if all_attn else None
        self.last_attention_weights = attn_arr
        return (
            np.array(all_labels),
            np.array(all_preds),
            np.vstack(all_probs),
            attn_arr,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> Dict:
        """Compute and return all evaluation metrics."""
        labels, preds, probs, attn_arr = self._collect_predictions()
        num_classes = probs.shape[1]

        accuracy        = float((labels == preds).mean())
        macro_precision = float(precision_score(labels, preds, average="macro", zero_division=0))
        macro_recall    = float(recall_score(labels, preds, average="macro", zero_division=0))
        macro_f1        = float(f1_score(labels, preds, average="macro", zero_division=0))
        weighted_f1     = float(f1_score(labels, preds, average="weighted", zero_division=0))

        top3_k   = min(3, num_classes)
        top3_acc = float(
            top_k_accuracy_score(labels, probs, k=top3_k, labels=list(range(num_classes)))
        )
        top5_k   = min(5, num_classes)
        top5_acc = float(
            top_k_accuracy_score(labels, probs, k=top5_k, labels=list(range(num_classes)))
        )

        label_names = [self.idx_to_label.get(i, str(i)) for i in range(num_classes)]
        clf_report  = classification_report(
            labels, preds,
            labels=list(range(num_classes)),
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))

        logger.info(
            "Top-1=%.4f | Top-%d=%.4f | Macro-F1=%.4f | Weighted-F1=%.4f",
            accuracy, top3_k, top3_acc, macro_f1, weighted_f1,
        )
        logger.info("\n%s", classification_report(
            labels, preds,
            labels=list(range(num_classes)),
            target_names=label_names,
            zero_division=0,
        ))

        return {
            "accuracy":            accuracy,
            "top3_accuracy":       top3_acc,
            f"top{top5_k}_accuracy": top5_acc,
            "macro_precision":     macro_precision,
            "macro_recall":        macro_recall,
            "macro_f1":            macro_f1,
            "weighted_f1":         weighted_f1,
            "per_class_report":    clf_report,
            "confusion_matrix":    cm.tolist(),
            "labels":              label_names,
            "attention_weights":   attn_arr,
        }

    def save_report(self, report: Dict, out_path: str | Path) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {k: v for k, v in report.items() if k != "attention_weights"}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        logger.info("Saved evaluation report → %s", out_path)

    @staticmethod
    def save_confusion_matrix_plot(
        report: Dict,
        plot_path: str | Path,
        dpi: int = 300,
    ) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = np.asarray(report["confusion_matrix"])
        labels = report.get("labels", [str(i) for i in range(cm.shape[0])])

        plot_path = Path(plot_path)
        plot_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax,
        )
        ax.set_xlabel("Predicted Gloss", fontsize=11)
        ax.set_ylabel("True Gloss", fontsize=11)
        ax.set_title(
            f"Confusion Matrix — ProposedResCNNBiLSTMAttn (Acc={report['accuracy']:.2%})",
            fontsize=12, fontweight="bold",
        )
        plt.xticks(rotation=35, ha="right")
        plt.yticks(rotation=0)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=dpi)
        plt.close(fig)
        logger.info("Saved confusion matrix plot → %s", plot_path)


# ─── Benchmark Plotting Helpers (300 DPI) ────────────────────────────────────

def _plot_benchmark_comparison(rows: List[Dict], out_path: Path, dpi: int = 300) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [r["Model Name"] for r in rows]
    top1 = [r["Top-1 Accuracy"] for r in rows]
    macro_f1 = [r["Macro F1-Score"] for r in rows]
    top3 = [r["Top-3 Accuracy"] for r in rows]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - width, top1, width, label="Top-1 Accuracy", color="#2563EB")
    b2 = ax.bar(x, macro_f1, width, label="Macro F1-Score", color="#10B981")
    b3 = ax.bar(x + width, top3, width, label="Top-3 Accuracy", color="#F59E0B")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=10)
    ax.set_ylim(0.0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Benchmark Comparison (Held-Out Signer-Disjoint Test Set)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_training_curves(
    histories: Dict[str, Dict[str, list]],
    out_path: Path,
    dpi: int = 300,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_loss, ax_acc = axes

    for name, hist in histories.items():
        val_loss = hist.get("val_loss", [])
        val_acc = hist.get("val_acc", [])
        epochs = np.arange(1, len(val_loss) + 1)
        if len(epochs) > 0:
            ax_loss.plot(epochs, val_loss, marker="o", markersize=3, linewidth=2, label=name)
            ax_acc.plot(epochs, val_acc, marker="o", markersize=3, linewidth=2, label=name)

    ax_loss.set_title("Validation Loss Across Epochs", fontsize=12, fontweight="bold")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Validation Loss")
    ax_loss.grid(True, linestyle="--", alpha=0.4)
    ax_loss.legend()

    ax_acc.set_title("Validation Accuracy Across Epochs", fontsize=12, fontweight="bold")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Validation Accuracy")
    ax_acc.set_ylim(0.0, 1.05)
    ax_acc.grid(True, linestyle="--", alpha=0.4)
    ax_acc.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_temporal_attention(
    attn_weights: np.ndarray,
    out_path: Path,
    max_samples: int = 8,
    dpi: int = 300,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = attn_weights.shape[1]
    frames = np.arange(T)
    mean_attn = attn_weights.mean(axis=0)
    std_attn = attn_weights.std(axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(min(max_samples, len(attn_weights))):
        ax.plot(frames, attn_weights[i], alpha=0.35, linewidth=1.2, label=f"Sample {i + 1}" if i < 4 else None)

    ax.plot(frames, mean_attn, color="#1D4ED8", linewidth=2.5, label="Mean Attention")
    ax.fill_between(
        frames,
        np.clip(mean_attn - std_attn, 0.0, 1.0),
        np.clip(mean_attn + std_attn, 0.0, 1.0),
        color="#93C5FD", alpha=0.35, label="±1 Std Dev",
    )

    ax.set_xticks(frames)
    ax.set_xlabel("Frame Index (0–29)", fontsize=11)
    ax.set_ylabel("Attention Weight", fontsize=11)
    ax.set_title("Temporal Self-Attention Weight Distribution — ProposedResCNNBiLSTMAttn", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# ─── Full 4-Model Benchmark Runner ───────────────────────────────────────────

def benchmark_all(
    config: Dict,
    test_loader: DataLoader,
    idx_to_label: Dict[int, str],
    checkpoints_dir: str | Path = "checkpoints",
    results_dir: str | Path = "results",
    device: str = "cpu",
    model_names: Optional[List[str]] = None,
    warmup_runs: int = 20,
    timed_runs: int = 100,
) -> Dict[str, object]:
    """Evaluate all 4 trained checkpoints and export CSVs, hardware specs, and 4 plots."""
    import gc
    checkpoints_path = Path(checkpoints_dir)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    names = model_names or ALL_MODEL_NAMES
    seq_len = int(config.get("sequence_length", 30))
    n_feat = int(config.get("num_features", 450))

    comparison_rows: List[Dict[str, object]] = []
    reports: Dict[str, Dict] = {}

    for name in names:
        ckpt_path = checkpoints_path / f"{name}_best.pt"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found for {name}: {ckpt_path}")

        model = load_model(str(ckpt_path), config, device=device, model_name=name)
        ev = Evaluator(model, test_loader, idx_to_label, device=device)
        report = ev.run()
        reports[name] = report

        param_count = int(model.count_parameters())  # type: ignore[operator]
        ckpt_size_kb = round(ckpt_path.stat().st_size / 1024.0, 2)
        lat_mean_ms, lat_std_ms = measure_cpu_latency(
            model,
            sequence_length=seq_len,
            num_features=n_feat,
            warmup_runs=warmup_runs,
            timed_runs=timed_runs,
        )

        comparison_rows.append({
            "Model Name": name,
            "Top-1 Accuracy": round(report["accuracy"], 4),
            "Top-3 Accuracy": round(report["top3_accuracy"], 4),
            "Macro Precision": round(report["macro_precision"], 4),
            "Macro Recall": round(report["macro_recall"], 4),
            "Macro F1-Score": round(report["macro_f1"], 4),
            "Weighted F1-Score": round(report["weighted_f1"], 4),
            "Parameter Count": param_count,
            "Checkpoint Size (KB)": ckpt_size_kb,
            "CPU Latency Mean (ms)": round(lat_mean_ms, 4),
            "CPU Latency Std (ms)": round(lat_std_ms, 4),
        })
        del model, ev
        gc.collect()

    # 1. Export results/model_comparison.csv
    comparison_csv = results_path / "model_comparison.csv"
    with open(comparison_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COMPARISON_COLUMNS)
        writer.writeheader()
        writer.writerows(comparison_rows)
    logger.info("Saved model comparison CSV → %s", comparison_csv)

    # 2. Export results/hardware_specs.json
    hw_specs = collect_hardware_specs()
    hw_path = results_path / "hardware_specs.json"
    with open(hw_path, "w", encoding="utf-8") as f:
        json.dump(hw_specs, f, indent=2)
    logger.info("Saved hardware specs → %s", hw_path)

    # 3. Export results/per_class_metrics.csv for ProposedResCNNBiLSTMAttn
    proposed_name = "ProposedResCNNBiLSTMAttn"
    proposed_report = reports[proposed_name]
    per_class_csv = results_path / "per_class_metrics.csv"
    clf_dict = proposed_report["per_class_report"]
    with open(per_class_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Class", "Precision", "Recall", "F1-Score", "Support"]
        )
        writer.writeheader()
        for cls_name in proposed_report["labels"]:
            metrics = clf_dict.get(cls_name, {})
            writer.writerow({
                "Class": cls_name,
                "Precision": round(float(metrics.get("precision", 0.0)), 4),
                "Recall": round(float(metrics.get("recall", 0.0)), 4),
                "F1-Score": round(float(metrics.get("f1-score", 0.0)), 4),
                "Support": int(metrics.get("support", 0)),
            })
    logger.info("Saved per-class metrics CSV → %s", per_class_csv)

    # 4. Generate 4 high-resolution (300 DPI) presentation charts
    _plot_benchmark_comparison(
        comparison_rows, results_path / "benchmark_comparison.png", dpi=300
    )

    history_file = results_path / "all_models_history.json"
    if history_file.is_file():
        with open(history_file, encoding="utf-8") as f:
            histories = json.load(f)
    else:
        histories = {}
    _plot_training_curves(
        histories, results_path / "training_curves.png", dpi=300
    )

    Evaluator.save_confusion_matrix_plot(
        proposed_report, results_path / "confusion_matrix_proposed.png", dpi=300
    )

    attn_weights = proposed_report.get("attention_weights")
    if attn_weights is not None:
        _plot_temporal_attention(
            attn_weights, results_path / "temporal_attention_weights.png", dpi=300
        )

    return {
        "comparison_rows": comparison_rows,
        "hardware_specs": hw_specs,
        "proposed_report": proposed_report,
    }


# ─── CLI entry point ─────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate & Benchmark Sign Language Models")
    p.add_argument("--config",          default="config.yaml")
    p.add_argument("--checkpoint",      default=None, help="Path to single checkpoint (non-benchmark mode)")
    p.add_argument("--checkpoints-dir", default="checkpoints", help="Checkpoints dir for --benchmark-all")
    p.add_argument("--results-dir",     default="results", help="Output dir for --benchmark-all")
    p.add_argument("--data",            default=None, help="WLASL .npz cache or feature dir")
    p.add_argument("--model",           default=None, help="Model name override (single-model mode)")
    p.add_argument("--out",             default="logs/eval_report.json")
    p.add_argument("--cm-plot",         default=None, help="Optional PNG path for single-model confusion matrix")
    p.add_argument("--benchmark-all",   action="store_true", help="Benchmark all 4 models in checkpoints/")
    p.add_argument("--device",          default="auto")
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

    data_source = args.data or config.get("cache_path", "data/wlasl10_landmarks.npz")
    _, _, test_loader, _, idx_to_label = build_dataloaders(
        data_source,
        batch_size=config.get("batch_size", 32),
        val_split=config.get("val_split", 0.15),
        test_split=config.get("test_split", 0.15),
        seed=config.get("seed", 42),
        wlasl_json_path=config.get("wlasl_json", "data/WLASL_v0.3.json"),
        raw_videos_dir=config.get("raw_videos_dir", "data/raw_videos"),
        num_classes=config.get("num_classes", 10),
        sequence_length=config.get("sequence_length", 30),
    )

    if args.benchmark_all:
        benchmark_all(
            config=config,
            test_loader=test_loader,
            idx_to_label=idx_to_label,
            checkpoints_dir=args.checkpoints_dir,
            results_dir=args.results_dir,
            device=device,
        )
        return

    if not args.checkpoint:
        raise ValueError("--checkpoint is required unless --benchmark-all is passed.")

    model = load_model(args.checkpoint, config, device=device, model_name=args.model)
    ev    = Evaluator(model, test_loader, idx_to_label, device=device)
    report = ev.run()
    ev.save_report(report, args.out)
    if args.cm_plot:
        ev.save_confusion_matrix_plot(report, args.cm_plot)


if __name__ == "__main__":
    main()
