"""
src/models.py
─────────────
Four PyTorch architectures for Sign Language Recognition on (Batch, 30, 450)
sequences alongside a unified ``build_model(name, ...)`` factory:

1. ``CNNBaseline``              : 1D-CNN spatial-temporal feature extractor + global avg/max pooling.
2. ``LSTMBaseline``             : 2-layer LSTM temporal sequence classifier.
3. ``CNNLSTMStandard``          : Standard Conv1D + MaxPool1D + LSTM hybrid.
4. ``ProposedResCNNBiLSTMAttn`` : Multi-scale Residual Conv1D (parallel k=3, k=5 + residual skip)
                                  -> Bidirectional LSTM -> Learnable Temporal Self-Attention,
                                  returning ``(logits, attention_weights)`` with
                                  ``attention_weights.shape == (Batch, 30)`` summing to 1.0.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─── Shared initialization mixin ─────────────────────────────────────────────

def _init_module_weights(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)


# ─── 1. CNNBaseline ──────────────────────────────────────────────────────────

class CNNBaseline(nn.Module):
    """1D-CNN feature extractor with global average + max pooling."""

    def __init__(
        self,
        num_features: int = 450,
        num_classes: int = 10,
        cnn_channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        dropout: float = 0.3,
        fc_dropout: float = 0.4,
    ) -> None:
        super().__init__()
        cnn_channels = cnn_channels or [128, 256]

        blocks: List[nn.Module] = []
        in_ch = num_features
        for out_ch in cnn_channels:
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                )
            )
            in_ch = out_ch

        self.conv = nn.Sequential(*blocks)
        pooled_dim = in_ch * 2  # concat(global_avg_pool, global_max_pool)
        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Dropout(fc_dropout),
            nn.Linear(pooled_dim, num_classes),
        )
        _init_module_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 450) -> logits: (B, num_classes)"""
        h = x.transpose(1, 2)               # (B, 450, T)
        h = self.conv(h)                    # (B, C, T)
        avg_pool = h.mean(dim=-1)           # (B, C)
        max_pool = h.amax(dim=-1)           # (B, C)
        feat = torch.cat([avg_pool, max_pool], dim=-1)  # (B, 2C)
        return self.classifier(feat)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── 2. LSTMBaseline ─────────────────────────────────────────────────────────

class LSTMBaseline(nn.Module):
    """2-layer LSTM temporal sequence classifier."""

    def __init__(
        self,
        num_features: int = 450,
        num_classes: int = 10,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.3,
        fc_dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_hidden_size),
            nn.Dropout(fc_dropout),
            nn.Linear(lstm_hidden_size, num_classes),
        )
        _init_module_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 450) -> logits: (B, num_classes)"""
        out, _ = self.lstm(x)               # (B, T, hidden)
        last = out[:, -1, :]                # (B, hidden)
        return self.classifier(last)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── 3. CNNLSTMStandard ──────────────────────────────────────────────────────

class CNNLSTMStandard(nn.Module):
    """Standard Conv1D + MaxPool1D + LSTM hybrid."""

    def __init__(
        self,
        num_features: int = 450,
        num_classes: int = 10,
        cnn_channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        cnn_dropout: float = 0.3,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.3,
        fc_dropout: float = 0.4,
    ) -> None:
        super().__init__()
        cnn_channels = cnn_channels or [128, 256]

        layers: List[nn.Module] = []
        in_ch = num_features
        for out_ch in cnn_channels:
            layers.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                    nn.Dropout(cnn_dropout),
                )
            )
            in_ch = out_ch

        self.cnn = nn.Sequential(*layers)
        self.lstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_hidden_size),
            nn.Dropout(fc_dropout),
            nn.Linear(lstm_hidden_size, num_classes),
        )
        _init_module_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 450) -> logits: (B, num_classes)"""
        h = x.transpose(1, 2)               # (B, 450, T)
        h = self.cnn(h)                     # (B, C, T')
        h = h.transpose(1, 2)               # (B, T', C)
        out, _ = self.lstm(h)               # (B, T', hidden)
        last = out[:, -1, :]                # (B, hidden)
        return self.classifier(last)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Backwards-compatible alias
SignLanguageCNNLSTM = CNNLSTMStandard


# ─── 4. ProposedResCNNBiLSTMAttn ─────────────────────────────────────────────

class MultiScaleResConv1DBlock(nn.Module):
    """Multi-scale Residual Conv1D block with parallel kernels (3 and 5) and skip connection.

    Preserves temporal sequence length T.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if out_channels % 2 != 0:
            raise ValueError(f"out_channels must be even for parallel k=3/k=5 split; got {out_channels}")
        branch_ch = out_channels // 2

        self.branch_k3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(branch_ch),
            nn.ReLU(inplace=True),
        )
        self.branch_k5 = nn.Sequential(
            nn.Conv1d(in_channels, branch_ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(branch_ch),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
            )
        )
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, T) -> (B, C_out, T)"""
        b3 = self.branch_k3(x)
        b5 = self.branch_k5(x)
        merged = torch.cat([b3, b5], dim=1)
        out = self.fuse(merged) + self.skip(x)
        return self.drop(self.act(out))


class TemporalSelfAttention(nn.Module):
    """Learnable Temporal Self-Attention pooling over T=30 time-steps."""

    def __init__(self, hidden_dim: int, attn_dim: Optional[int] = None) -> None:
        super().__init__()
        attn_dim = attn_dim or max(64, hidden_dim // 2)
        self.query_proj = nn.Linear(hidden_dim, attn_dim)
        self.score_vec = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """h: (B, T, D) -> context: (B, D), attention_weights: (B, T) summing to 1.0."""
        energy = self.score_vec(torch.tanh(self.query_proj(h))).squeeze(-1)  # (B, T)
        attn_weights = F.softmax(energy, dim=1)                              # (B, T)
        context = torch.bmm(attn_weights.unsqueeze(1), h).squeeze(1)         # (B, D)
        return context, attn_weights


class ProposedResCNNBiLSTMAttn(nn.Module):
    """Multi-scale Residual Conv1D -> Bidirectional LSTM -> Temporal Self-Attention.

    Returns ``(logits, attention_weights)`` where ``attention_weights`` has shape
    ``(Batch, 30)`` and sums to 1.0 across time.
    """

    def __init__(
        self,
        num_features: int = 450,
        num_classes: int = 10,
        cnn_channels: Optional[List[int]] = None,
        cnn_dropout: float = 0.3,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.3,
        fc_dropout: float = 0.4,
    ) -> None:
        super().__init__()
        cnn_channels = cnn_channels or [128, 256]

        res_blocks: List[nn.Module] = []
        in_ch = num_features
        for out_ch in cnn_channels:
            res_blocks.append(
                MultiScaleResConv1DBlock(in_ch, out_ch, dropout=cnn_dropout)
            )
            in_ch = out_ch
        self.res_cnn = nn.Sequential(*res_blocks)

        self.bilstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
            bidirectional=True,
        )

        bilstm_dim = lstm_hidden_size * 2
        self.attention = TemporalSelfAttention(bilstm_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(bilstm_dim),
            nn.Dropout(fc_dropout),
            nn.Linear(bilstm_dim, num_classes),
        )
        _init_module_weights(self)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, 30, 450) -> (logits: (B, num_classes), attention_weights: (B, 30))."""
        h = x.transpose(1, 2)                           # (B, 450, 30)
        h = self.res_cnn(h)                             # (B, C, 30)
        h = h.transpose(1, 2)                           # (B, 30, C)
        seq_out, _ = self.bilstm(h)                     # (B, 30, 2*hidden)
        context, attn_weights = self.attention(seq_out) # (B, 2*hidden), (B, 30)
        logits = self.classifier(context)               # (B, num_classes)
        return logits, attn_weights

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Unified Factory ─────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "cnnbaseline": CNNBaseline,
    "cnn_baseline": CNNBaseline,
    "lstmbaseline": LSTMBaseline,
    "lstm_baseline": LSTMBaseline,
    "cnnlstmstandard": CNNLSTMStandard,
    "cnn_lstm_standard": CNNLSTMStandard,
    "signlanguagecnnlstm": CNNLSTMStandard,
    "proposedrescnnbilstmattn": ProposedResCNNBiLSTMAttn,
    "proposed_res_cnn_bilstm_attn": ProposedResCNNBiLSTMAttn,
    "proposed": ProposedResCNNBiLSTMAttn,
}


def build_model(
    name_or_config: Union[str, Dict[str, Any]],
    **kwargs: Any,
) -> nn.Module:
    """Unified factory for all 4 Sign Language Recognition architectures.

    Supports both:
        build_model("ProposedResCNNBiLSTMAttn", num_features=450, num_classes=10, ...)
    and config-dict invocation:
        build_model(config_dict)
    """
    if isinstance(name_or_config, dict):
        cfg = dict(name_or_config)
        name = str(cfg.get("model_name", "CNNLSTMStandard"))
        merged: Dict[str, Any] = {
            "num_features": cfg.get("num_features", 450),
            "num_classes": cfg.get("num_classes", 10),
            "cnn_channels": cfg.get("cnn_out_channels", cfg.get("cnn_channels", [128, 256])),
            "kernel_size": cfg.get("cnn_kernel_size", 3),
            "cnn_dropout": cfg.get("cnn_dropout", 0.3),
            "dropout": cfg.get("cnn_dropout", 0.3),
            "lstm_hidden_size": cfg.get("lstm_hidden_size", 256),
            "lstm_num_layers": cfg.get("lstm_num_layers", 2),
            "lstm_dropout": cfg.get("lstm_dropout", 0.3),
            "fc_dropout": cfg.get("fc_dropout", 0.4),
        }
        merged.update(kwargs)
    else:
        name = str(name_or_config)
        merged = dict(kwargs)

    key = name.strip().lower().replace("-", "_")
    if key not in MODEL_REGISTRY:
        available = ["CNNBaseline", "LSTMBaseline", "CNNLSTMStandard", "ProposedResCNNBiLSTMAttn"]
        raise ValueError(f"Unknown model {name!r}. Available: {available}")

    cls = MODEL_REGISTRY[key]
    # Filter kwargs to those accepted by the chosen model's __init__
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    filtered_kwargs = {k: v for k, v in merged.items() if k in valid_params}

    model = cls(**filtered_kwargs)
    logger.info("Built model %s (%d trainable params)", cls.__name__, model.count_parameters())
    return model


def load_model(
    checkpoint_path: str,
    config: dict,
    device: str = "cpu",
    model_name: Optional[str] = None,
) -> nn.Module:
    """Load a model from a saved checkpoint."""
    state = torch.load(checkpoint_path, map_location=device)
    cfg = state.get("config", config) if isinstance(state, dict) else config
    name = model_name or cfg.get("model_name", "CNNLSTMStandard")
    model = build_model(name, **{
        "num_features": cfg.get("num_features", 450),
        "num_classes": cfg.get("num_classes", 10),
        "cnn_channels": cfg.get("cnn_out_channels", cfg.get("cnn_channels", [128, 256])),
        "kernel_size": cfg.get("cnn_kernel_size", 3),
        "cnn_dropout": cfg.get("cnn_dropout", 0.3),
        "dropout": cfg.get("cnn_dropout", 0.3),
        "lstm_hidden_size": cfg.get("lstm_hidden_size", 256),
        "lstm_num_layers": cfg.get("lstm_num_layers", 2),
        "lstm_dropout": cfg.get("lstm_dropout", 0.3),
        "fc_dropout": cfg.get("fc_dropout", 0.4),
    })
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info("Loaded %s from %s", model.__class__.__name__, checkpoint_path)
    return model
