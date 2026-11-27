"""
src/train.py
────────────
Training loop with early stopping, LR scheduling, checkpoint saving, and
sequential 4-model benchmarking via ``--train-all``.

Usage (command line)
────────────────────
    # Train a single model
    python -m src.train --config config.yaml --data data/wlasl10_landmarks.npz --out models/

    # Train all 4 models on the exact same signer-disjoint split
    python -m src.train --config config.yaml --data data/wlasl10_landmarks.npz --train-all

    # Fast isolated smoke test
    python -m src.train --config config.yaml --model ProposedResCNNBiLSTMAttn --smoke-test
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import build_dataloaders, generate_smoke_test_data
from src.models import build_model

logger = logging.getLogger(__name__)

ALL_MODEL_NAMES: List[str] = [
    "CNNBaseline",
    "LSTMBaseline",
    "CNNLSTMStandard",
    "ProposedResCNNBiLSTMAttn",
]


# ─── Seed ────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── Trainer ─────────────────────────────────────────────────────────────────

class Trainer:
    """Encapsulates the full training procedure for any of the 4 models.

    Parameters
    ----------
    config : dict
        Parsed config.yaml as a Python dict.
    train_loader, val_loader : DataLoader
        Training and validation data loaders.
    device : str
        "cuda", "mps", or "cpu".
    out_dir : str | Path
        Directory to save checkpoints and training history.
    idx_to_label : dict, optional
        Class index -> gloss name mapping stored in checkpoints.
    label_to_idx : dict, optional
        Gloss name -> class index mapping stored in checkpoints.
    checkpoint_name : str
        Filename for the saved best checkpoint (default ``"best_model.pt"``).
    save_history : bool
        Whether to write ``history.json`` in ``out_dir``.
    """

    def __init__(
        self,
        config: Dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cpu",
        out_dir: str | Path = "models",
        idx_to_label: Optional[Dict[int, str]] = None,
        label_to_idx: Optional[Dict[str, int]] = None,
        checkpoint_name: str = "best_model.pt",
        save_history: bool = True,
    ) -> None:
        self.config          = dict(config)
        self.train_loader    = train_loader
        self.val_loader      = val_loader
        self.device          = torch.device(device)
        self.out_dir         = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.idx_to_label    = idx_to_label
        self.label_to_idx    = label_to_idx
        self.checkpoint_name = checkpoint_name
        self.save_history    = save_history

        set_seed(self.config.get("seed", 42))

        self.model: nn.Module = build_model(self.config).to(self.device)
        logger.info(
            "Model %s parameters: %d",
            self.model.__class__.__name__,
            self.model.count_parameters(),  # type: ignore[operator]
        )

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 1e-3),
            weight_decay=self.config.get("weight_decay", 1e-4),
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self.config.get("num_epochs", 50)),
            eta_min=1e-6,
        )

        self.num_epochs     = self.config.get("num_epochs", 50)
        self.patience       = self.config.get("early_stopping_patience", 10)
        self.grad_clip      = self.config.get("grad_clip", 1.0)

        self.best_val_loss  = float("inf")
        self.patience_count = 0
        self.history: Dict[str, list] = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
        }

    # ── Epoch helpers ─────────────────────────────────────────────────────────

    def _run_epoch(
        self, loader: DataLoader, training: bool
    ) -> Tuple[float, float]:
        self.model.train(training)
        total_loss = 0.0
        correct    = 0
        total      = 0

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for x, y in tqdm(loader, desc="train" if training else "val", leave=False):
                x, y = x.to(self.device), y.to(self.device)

                if training:
                    self.optimizer.zero_grad()

                out = self.model(x)
                logits = out[0] if isinstance(out, tuple) else out
                loss   = self.criterion(logits, y)

                if training:
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                    self.optimizer.step()

                total_loss += loss.item() * x.size(0)
                preds      = logits.argmax(dim=1)
                correct   += (preds == y).sum().item()
                total     += x.size(0)

        return total_loss / max(1, total), correct / max(1, total)

    # ── Save / load ───────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, val_loss: float) -> Path:
        path = self.out_dir / self.checkpoint_name
        torch.save(
            {
                "epoch": epoch,
                "model_name": self.model.__class__.__name__,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "config": self.config,
                "idx_to_label": self.idx_to_label,
                "label_to_idx": self.label_to_idx,
            },
            path,
        )
        logger.info("  ✓ Saved checkpoint → %s  (val_loss=%.4f)", path, val_loss)
        return path

    # ── Main fit loop ─────────────────────────────────────────────────────────

    def fit(self) -> Dict[str, list]:
        """Run the full training loop and optionally persist ``history.json``."""
        logger.info("Starting training for %d epochs on %s", self.num_epochs, self.device)
        t0 = time.time()

        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_acc = self._run_epoch(self.train_loader, training=True)
            val_loss,   val_acc   = self._run_epoch(self.val_loader,   training=False)
            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                "Epoch %3d/%d  |  "
                "train loss=%.4f acc=%.3f  |  "
                "val loss=%.4f acc=%.3f  |  lr=%.2e",
                epoch, self.num_epochs,
                train_loss, train_acc,
                val_loss, val_acc, lr,
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss  = val_loss
                self.patience_count = 0
                self._save_checkpoint(epoch, val_loss)
            else:
                self.patience_count += 1
                if self.patience_count >= self.patience:
                    logger.info(
                        "Early stopping at epoch %d (patience=%d)",
                        epoch, self.patience,
                    )
                    break

        elapsed = time.time() - t0
        if self.save_history:
            history_path = self.out_dir / "history.json"
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)
            logger.info("Training complete in %.1f s (history → %s)", elapsed, history_path)
        else:
            logger.info("Training complete in %.1f s", elapsed)
        return self.history


# ─── Multi-model training helper ─────────────────────────────────────────────

def train_all_models(
    config: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = "cpu",
    checkpoints_dir: str | Path = "checkpoints",
    results_dir: str | Path = "results",
    idx_to_label: Optional[Dict[int, str]] = None,
    label_to_idx: Optional[Dict[str, int]] = None,
    model_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, list]]:
    """Sequentially train all 4 models on the exact same signer-disjoint split.

    Saves:
        - ``<checkpoints_dir>/<model_name>_best.pt`` for each model
        - ``<results_dir>/all_models_history.json`` with combined epoch histories
    """
    import gc
    checkpoints_path = Path(checkpoints_dir)
    results_path = Path(results_dir)
    checkpoints_path.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)

    names = model_names or ALL_MODEL_NAMES
    all_histories: Dict[str, Dict[str, list]] = {}

    for model_name in names:
        logger.info("══════════════════════════════════════════════════════")
        logger.info("Training %s (seed=%s)", model_name, config.get("seed", 42))
        logger.info("══════════════════════════════════════════════════════")
        model_cfg = dict(config)
        model_cfg["model_name"] = model_name

        trainer = Trainer(
            model_cfg,
            train_loader,
            val_loader,
            device=device,
            out_dir=checkpoints_path,
            idx_to_label=idx_to_label,
            label_to_idx=label_to_idx,
            checkpoint_name=f"{model_name}_best.pt",
            save_history=False,
        )
        all_histories[model_name] = trainer.fit()
        del trainer
        gc.collect()

    history_file = results_path / "all_models_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(all_histories, f, indent=2)
    logger.info("Saved combined epoch histories → %s", history_file)

    return all_histories


# ─── CLI entry point ─────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Sign Language Recognition Model(s)")
    p.add_argument("--config",          default="config.yaml")
    p.add_argument("--data",            default=None, help="WLASL .npz cache or feature dir")
    p.add_argument("--out",             default="models", help="Checkpoint output dir (single-model mode)")
    p.add_argument("--checkpoints-dir", default="checkpoints", help="Checkpoint dir for --train-all")
    p.add_argument("--results-dir",     default="results", help="Results dir for --train-all history")
    p.add_argument("--model",           default=None, help="Model name override (e.g. ProposedResCNNBiLSTMAttn)")
    p.add_argument("--epochs",          type=int, default=None, help="Epoch count override")
    p.add_argument("--device",          default="auto")
    p.add_argument("--train-all",       action="store_true", help="Sequentially train all 4 models on the same split")
    p.add_argument("--smoke-test",      action="store_true", help="Run fast isolated 2-epoch training smoke test")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    args = _parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.model is not None:
        config["model_name"] = args.model
    if args.epochs is not None:
        config["num_epochs"] = args.epochs

    device = args.device
    if device == "auto":
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    logger.info("Using device: %s", device)

    pin = device.startswith("cuda")

    if args.smoke_test:
        config["num_epochs"] = min(config.get("num_epochs", 2), 2)
        smoke = generate_smoke_test_data(
            num_classes=config.get("num_classes", 10),
            sequence_length=config.get("sequence_length", 30),
            num_features=config.get("num_features", 450),
            seed=config.get("seed", 42),
        )
        train_loader, val_loader, _, label_to_idx, idx_to_label = build_dataloaders(
            batch_size=config.get("batch_size", 32),
            val_split=config.get("val_split", 0.15),
            test_split=config.get("test_split", 0.15),
            seed=config.get("seed", 42),
            pin_memory=pin,
            X=smoke["X"],                  # type: ignore[arg-type]
            y=smoke["y"],                  # type: ignore[arg-type]
            signer_ids=smoke["signer_id"],  # type: ignore[arg-type]
            label_to_idx=smoke["label_to_idx"],  # type: ignore[arg-type]
            idx_to_label=smoke["idx_to_label"],  # type: ignore[arg-type]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            if args.train_all:
                train_all_models(
                    config, train_loader, val_loader,
                    device=device,
                    checkpoints_dir=Path(tmp_dir) / "checkpoints",
                    results_dir=Path(tmp_dir) / "results",
                    idx_to_label=idx_to_label,
                    label_to_idx=label_to_idx,
                )
            else:
                trainer = Trainer(
                    config, train_loader, val_loader,
                    device=device, out_dir=tmp_dir,
                    idx_to_label=idx_to_label, label_to_idx=label_to_idx,
                )
                trainer.fit()
        return

    data_source = args.data or config.get("cache_path", "data/wlasl10_landmarks.npz")
    train_loader, val_loader, _, label_to_idx, idx_to_label = build_dataloaders(
        data_source,
        batch_size=config.get("batch_size", 32),
        val_split=config.get("val_split", 0.15),
        test_split=config.get("test_split", 0.15),
        seed=config.get("seed", 42),
        pin_memory=pin,
        wlasl_json_path=config.get("wlasl_json", "data/WLASL_v0.3.json"),
        raw_videos_dir=config.get("raw_videos_dir", "data/raw_videos"),
        num_classes=config.get("num_classes", 10),
        sequence_length=config.get("sequence_length", 30),
    )

    if args.train_all:
        train_all_models(
            config, train_loader, val_loader,
            device=device,
            checkpoints_dir=args.checkpoints_dir,
            results_dir=args.results_dir,
            idx_to_label=idx_to_label,
            label_to_idx=label_to_idx,
        )
    else:
        trainer = Trainer(
            config, train_loader, val_loader,
            device=device, out_dir=args.out,
            idx_to_label=idx_to_label, label_to_idx=label_to_idx,
        )
        trainer.fit()


if __name__ == "__main__":
    main()
