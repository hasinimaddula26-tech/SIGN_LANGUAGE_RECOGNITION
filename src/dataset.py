"""
src/dataset.py
──────────────
Dataset preparation, caching, signer-aware stratified splitting, kinematic
augmentation, and PyTorch DataLoaders for Sign Language Recognition.

Supported data layouts
──────────────────────
1. Official WLASL layout (primary):
   • Flat video directory : data/raw_videos/<video_id>.mp4
   • Metadata JSON        : data/WLASL_v0.3.json
     (list of {"gloss": str, "instances": [{"video_id", "signer_id", "split",
                                            "frame_start", "frame_end", ...}]})
   • Landmark cache       : data/wlasl10_landmarks.npz
     (top-10 glosses by available .mp4 count, features (N, 30, 450), gloss, signer_id)

2. Class-subfolder fallback layout:
   • data/features/<class_name>/<sample_id>.npy  # shape (30, 450), float32

CLI usage
─────────
    # Extract & cache WLASL-10 landmarks
    python -m src.dataset --wlasl-json data/WLASL_v0.3.json --raw-videos data/raw_videos

    # Isolated in-memory smoke test (never touches data/wlasl10_landmarks.npz)
    python -m src.dataset --smoke-test-data
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.features import LandmarkExtractor, N_COORDS, N_FEATURES, _compute_velocity, _resample

logger = logging.getLogger(__name__)


# ─── Dataset ─────────────────────────────────────────────────────────────────

_POSE_SWAP_PAIRS: Tuple[Tuple[int, int], ...] = (
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12),
    (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
)


class SignLanguageDataset(Dataset):
    """PyTorch Dataset supporting cached in-memory arrays or .npy file paths.

    Realistic kinematic & domain-shift augmentation (temporal crop, speed
    resampling ±15%, horizontal left/right mirroring, spatial scale/shift
    jitter, and Gaussian joint noise std=0.015) is applied ONLY when
    ``augment=True`` (training split only — never validation or test).
    """

    def __init__(
        self,
        samples: Optional[List[Tuple[Path, int]]] = None,
        label_to_idx: Optional[Dict[str, int]] = None,
        augment: bool = False,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        jitter_std: float = 0.015,
        speed_range: Tuple[float, float] = (0.85, 1.15),
        crop_range: Tuple[float, float] = (0.80, 1.00),
        scale_range: Tuple[float, float] = (0.90, 1.10),
        shift_range: Tuple[float, float] = (-0.03, 0.03),
        mirror_prob: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        if X is not None and y is not None:
            if len(X) != len(y):
                raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
            self.X = np.asarray(X, dtype=np.float32)
            self.y = np.asarray(y, dtype=np.int64)
            self.samples = None
        elif samples is not None:
            self.samples = samples
            self.X = None
            self.y = None
        else:
            raise ValueError("Provide either (X, y) arrays or samples list.")

        self.label_to_idx = label_to_idx or {}
        self.augment = augment
        self.jitter_std = jitter_std
        self.speed_range = speed_range
        self.crop_range = crop_range
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.mirror_prob = mirror_prob
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.X) if self.X is not None else len(self.samples)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.X is not None and self.y is not None:
            feat = self.X[idx].copy()
            label = int(self.y[idx])
        else:
            path, label = self.samples[idx]  # type: ignore[index]
            feat = np.load(str(path)).astype(np.float32)

        if self.augment:
            feat = self._augment(
                feat,
                rng=self._rng,
                speed_range=self.speed_range,
                jitter_std=self.jitter_std,
                crop_range=self.crop_range,
                scale_range=self.scale_range,
                shift_range=self.shift_range,
                mirror_prob=self.mirror_prob,
            )

        x = torch.from_numpy(np.ascontiguousarray(feat, dtype=np.float32))
        y = torch.tensor(label, dtype=torch.long)
        return x, y

    # ── Kinematic & Domain-Shift Augmentation (Train Only) ───────────────────

    @staticmethod
    def _augment(
        feat: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        speed_range: Tuple[float, float] = (0.85, 1.15),
        jitter_std: float = 0.015,
        crop_range: Tuple[float, float] = (0.80, 1.00),
        scale_range: Tuple[float, float] = (0.90, 1.10),
        shift_range: Tuple[float, float] = (-0.03, 0.03),
        mirror_prob: float = 0.5,
    ) -> np.ndarray:
        """Apply webcam-robust temporal + spatial augmentations to training clips.

        1. Random temporal crop (80%–100% contiguous window) resampled back to T.
        2. Random temporal speed resampling (±15%).
        3. Random horizontal left/right mirroring (negates x, swaps L/R pose & hands).
        4. Random spatial scale jitter (0.90x–1.10x) & shoulder-center shift (±0.03).
        5. Gaussian joint jitter (std=0.015) on detected (non-zero) joints.
        6. Recompute consistent velocity ``vel = Δcoords`` (with ``vel[0] = 0``).
        """
        if rng is None:
            rng = np.random.default_rng()

        T, D = feat.shape
        n_coords = min(N_COORDS, D // 2) if D >= 2 else D
        coords = feat[:, :n_coords].astype(np.float32, copy=True)

        # 1. Random temporal crop (80%–100% contiguous window) resampled to T
        crop_ratio = float(rng.uniform(crop_range[0], crop_range[1]))
        crop_len = max(2, min(T, int(round(T * crop_ratio))))
        if crop_len < T:
            start = int(rng.integers(0, T - crop_len + 1))
            cropped = coords[start : start + crop_len]
            src_t = np.arange(crop_len, dtype=np.float32)
            dst_t = np.linspace(0.0, crop_len - 1, T, dtype=np.float32)
            resampled = np.empty((T, n_coords), dtype=np.float32)
            for col in range(n_coords):
                resampled[:, col] = np.interp(dst_t, src_t, cropped[:, col])
            coords = resampled

        # 2. Random temporal speed resampling (±15%)
        speed = float(rng.uniform(speed_range[0], speed_range[1]))
        t_new = max(2, int(round(T / speed)))
        if t_new != T:
            src_t = np.arange(T, dtype=np.float32)
            dst_t = np.linspace(0.0, T - 1, t_new, dtype=np.float32)
            warped = np.empty((t_new, n_coords), dtype=np.float32)
            for col in range(n_coords):
                warped[:, col] = np.interp(dst_t, src_t, coords[:, col])
            coords = _resample(warped, target=T)

        # 3. Horizontal mirror invariance (negate x, swap symmetric L/R pose + L/R hands)
        if mirror_prob > 0.0 and rng.random() < mirror_prob and n_coords == N_COORDS:
            kp_m = coords.reshape(T, 75, 3)
            det_m = np.any(kp_m != 0.0, axis=-1)
            kp_m[:, :, 0] = np.where(det_m, -kp_m[:, :, 0], 0.0)
            for idx_l, idx_r in _POSE_SWAP_PAIRS:
                kp_m[:, [idx_l, idx_r]] = kp_m[:, [idx_r, idx_l]]
            left_hand = kp_m[:, 33:54].copy()
            kp_m[:, 33:54] = kp_m[:, 54:75]
            kp_m[:, 54:75] = left_hand
            coords = kp_m.reshape(T, n_coords)

        # 4. Spatial scale jitter (0.90x–1.10x), shoulder-center shift (±0.03), & joint noise
        scale = float(rng.uniform(scale_range[0], scale_range[1]))
        shift = rng.uniform(shift_range[0], shift_range[1], size=(1, 1, 3)).astype(np.float32)
        if n_coords % 3 == 0:
            kp = coords.reshape(T, n_coords // 3, 3)
            detected = np.any(kp != 0.0, axis=-1, keepdims=True)
            kp = kp * scale + shift
            if jitter_std > 0.0:
                kp = kp + rng.normal(0.0, jitter_std, size=kp.shape).astype(np.float32)
            kp = kp * detected
            coords = kp.reshape(T, n_coords)
        else:
            detected = coords != 0.0
            coords = coords * scale + float(shift[0, 0, 0])
            if jitter_std > 0.0:
                coords = coords + rng.normal(0.0, jitter_std, size=coords.shape).astype(np.float32)
            coords = coords * detected

        # 4. Recompute consistent velocity along temporal axis
        if D == n_coords * 2:
            vel = _compute_velocity(coords)
            out = np.concatenate([coords, vel], axis=1)
        else:
            out = coords

        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# ─── WLASL Preparation & Caching ─────────────────────────────────────────────

def select_top_wlasl_glosses(
    wlasl_json_path: str | Path,
    raw_videos_dir: str | Path,
    num_classes: int = 10,
) -> List[Tuple[str, List[Dict]]]:
    """Filter WLASL_v0.3.json for the top *num_classes* glosses by available .mp4 count.

    Returns a list of ``(gloss, valid_instances)`` sorted deterministically by
    descending count (ties broken alphabetically by gloss).
    """
    wlasl_json_path = Path(wlasl_json_path)
    raw_videos_dir = Path(raw_videos_dir)

    if not wlasl_json_path.is_file():
        raise FileNotFoundError(f"WLASL metadata JSON not found: {wlasl_json_path}")
    if not raw_videos_dir.is_dir():
        raise FileNotFoundError(f"WLASL raw videos directory not found: {raw_videos_dir}")

    with open(wlasl_json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    candidates: List[Tuple[str, List[Dict]]] = []
    for entry in entries:
        gloss = entry.get("gloss", "").strip()
        if not gloss:
            continue
        valid_instances: List[Dict] = []
        for inst in entry.get("instances", []):
            vid_id = str(inst.get("video_id", "")).strip()
            if not vid_id:
                continue
            vid_path = raw_videos_dir / f"{vid_id}.mp4"
            if vid_path.is_file():
                inst_copy = dict(inst)
                inst_copy["video_path"] = str(vid_path)
                valid_instances.append(inst_copy)
        if valid_instances:
            candidates.append((gloss, valid_instances))

    if not candidates:
        raise ValueError(
            f"No matching .mp4 videos from {wlasl_json_path} found in {raw_videos_dir}"
        )

    candidates.sort(
        key=lambda item: (
            -len(item[1]),
            -len({inst.get("signer_id") for inst in item[1]}),
            item[0],
        )
    )
    selected = candidates[:num_classes]
    logger.info(
        "Selected top %d WLASL glosses: %s",
        len(selected),
        [(g, len(insts), len({i.get("signer_id") for i in insts})) for g, insts in selected],
    )
    return selected


_WORKER_EXTRACTOR: Optional[LandmarkExtractor] = None


def _init_worker_extractor(sequence_length: int) -> None:
    global _WORKER_EXTRACTOR
    _WORKER_EXTRACTOR = LandmarkExtractor(sequence_length=sequence_length)


def _extract_clip_task(
    task: Tuple[str, str, int, str, int, Optional[int], Optional[int], str]
) -> Optional[Tuple[str, int, str, int, str]]:
    """Worker helper: extract a single clip and save to per-clip cache .npy."""
    vid_path, vid_id, lbl, gloss, signer_id, frame_start, frame_end, clip_file_str = task
    clip_file = Path(clip_file_str)
    if clip_file.is_file():
        return clip_file_str, lbl, gloss, signer_id, vid_id

    global _WORKER_EXTRACTOR
    if _WORKER_EXTRACTOR is None:
        return None
    try:
        feat = _WORKER_EXTRACTOR.process_video(
            vid_path,
            frame_start=frame_start,
            frame_end=frame_end,
        )
        clip_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(clip_file, feat)
        return clip_file_str, lbl, gloss, signer_id, vid_id
    except Exception as exc:
        logger.warning("Skipping %s (%s): %s", vid_id, vid_path, exc)
        return None


def prepare_wlasl_dataset(
    wlasl_json_path: str | Path = "data/WLASL_v0.3.json",
    raw_videos_dir: str | Path = "data/raw_videos",
    cache_path: str | Path = "data/wlasl10_landmarks.npz",
    num_classes: int = 10,
    sequence_length: int = 30,
    force_recompute: bool = False,
    num_workers: int = 1,
) -> Dict[str, object]:
    """Extract WLASL top-N gloss landmarks and cache to ``.npz`` (or load existing cache).

    Uses per-clip intermediate caching for resumability and parallel extraction
    when ``num_workers > 1`` and multiple clips remain.
    """
    import os
    import shutil
    from concurrent.futures import ProcessPoolExecutor

    cache_path = Path(cache_path)
    if cache_path.is_file() and not force_recompute:
        return load_wlasl_cache(cache_path)

    selected = select_top_wlasl_glosses(
        wlasl_json_path=wlasl_json_path,
        raw_videos_dir=raw_videos_dir,
        num_classes=num_classes,
    )

    class_names = sorted(gloss for gloss, _ in selected)
    label_to_idx = {name: i for i, name in enumerate(class_names)}
    idx_to_label = {i: name for i, name in enumerate(class_names)}

    clip_cache_dir = cache_path.parent / f".{cache_path.stem}_clips"
    if force_recompute and clip_cache_dir.exists():
        shutil.rmtree(clip_cache_dir, ignore_errors=True)
    clip_cache_dir.mkdir(parents=True, exist_ok=True)

    tasks: List[Tuple[str, str, int, str, int, Optional[int], Optional[int], str]] = []
    for gloss, instances in selected:
        lbl = label_to_idx[gloss]
        for inst in instances:
            vid_path = inst["video_path"]
            vid_id = str(inst["video_id"])
            signer_id = int(inst.get("signer_id", -1))
            fs = inst.get("frame_start")
            fe = inst.get("frame_end")
            clip_file = clip_cache_dir / f"{gloss}_{vid_id}.npy"
            tasks.append((
                vid_path,
                vid_id,
                lbl,
                gloss,
                signer_id,
                int(fs) if fs is not None else None,
                int(fe) if fe is not None else None,
                str(clip_file),
            ))

    pending = [t for t in tasks if not Path(t[7]).is_file()]
    logger.info("Total clips: %d | Already cached: %d | To extract: %d", len(tasks), len(tasks) - len(pending), len(pending))

    if pending:
        workers = min(num_workers, len(pending), os.cpu_count() or 1)
        if workers > 1 and len(pending) >= 8:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker_extractor,
                initargs=(sequence_length,),
            ) as pool:
                list(pool.map(_extract_clip_task, pending))
        else:
            with LandmarkExtractor(sequence_length=sequence_length) as extractor:
                for idx, (vid_path, vid_id, lbl, gloss, signer_id, fs, fe, clip_file_str) in enumerate(pending, start=1):
                    try:
                        feat = extractor.process_video(vid_path, frame_start=fs, frame_end=fe)
                        np.save(clip_file_str, feat)
                        if idx % 10 == 0 or idx == len(pending):
                            logger.info("Extracted %d/%d pending clips (%s:%s)", idx, len(pending), gloss, vid_id)
                    except Exception as exc:
                        logger.warning("Skipping %s (%s): %s", vid_id, vid_path, exc)

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    gloss_list: List[str] = []
    signer_list: List[int] = []
    video_id_list: List[str] = []

    for _, vid_id, lbl, gloss, signer_id, _, _, clip_file_str in tasks:
        clip_p = Path(clip_file_str)
        if clip_p.is_file():
            X_list.append(np.load(clip_p).astype(np.float32))
            y_list.append(lbl)
            gloss_list.append(gloss)
            signer_list.append(signer_id)
            video_id_list.append(vid_id)

    if not X_list:
        raise RuntimeError("No videos could be processed from WLASL dataset.")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    gloss_arr = np.asarray(gloss_list)
    signer_arr = np.asarray(signer_list, dtype=np.int64)
    video_id_arr = np.asarray(video_id_list)
    classes_arr = np.asarray(class_names)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        X=X,
        y=y,
        gloss=gloss_arr,
        signer_id=signer_arr,
        video_id=video_id_arr,
        classes=classes_arr,
    )
    shutil.rmtree(clip_cache_dir, ignore_errors=True)
    logger.info("Cached %d WLASL samples (%d classes) → %s", len(X), len(class_names), cache_path)

    return {
        "X": X,
        "y": y,
        "gloss": gloss_arr,
        "signer_id": signer_arr,
        "video_id": video_id_arr,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
    }


def load_wlasl_cache(cache_path: str | Path = "data/wlasl10_landmarks.npz") -> Dict[str, object]:
    """Load cached WLASL landmark features, gloss labels, and signer IDs from ``.npz``."""
    cache_path = Path(cache_path)
    if not cache_path.is_file():
        raise FileNotFoundError(f"Landmark cache not found: {cache_path}")

    with np.load(cache_path, allow_pickle=False) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)
        gloss = data["gloss"]
        signer_id = data["signer_id"].astype(np.int64)
        video_id = data["video_id"] if "video_id" in data else np.arange(len(y)).astype(str)
        if "classes" in data:
            classes = [str(c) for c in data["classes"]]
        else:
            classes = sorted({str(g) for g in gloss})

    label_to_idx = {name: i for i, name in enumerate(classes)}
    idx_to_label = {i: name for i, name in enumerate(classes)}
    logger.info("Loaded %d cached samples across %d classes from %s", len(X), len(classes), cache_path)

    return {
        "X": X,
        "y": y,
        "gloss": gloss,
        "signer_id": signer_id,
        "video_id": video_id,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
    }


# ─── Signer-Aware Stratified Splitting ───────────────────────────────────────

def _choose_fold_subset(
    folds: List[np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    target_ratio: float,
    min_remaining_groups: int = 1,
    min_chosen_groups: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select a subset of StratifiedGroupKFold folds closest to *target_ratio*."""
    n_total = len(y)
    n_folds = len(folds)
    all_classes = set(np.unique(y).tolist())

    best_score = float("inf")
    best_chosen: Optional[Tuple[int, ...]] = None

    # Search fold combinations (n_folds <= 10 -> at most 1022 combinations)
    for k in range(1, n_folds):
        for combo in itertools.combinations(range(n_folds), k):
            chosen_idx = np.concatenate([folds[i] for i in combo])
            rem_idx = np.concatenate([folds[i] for i in range(n_folds) if i not in combo])

            chosen_groups = len(np.unique(groups[chosen_idx]))
            rem_groups = len(np.unique(groups[rem_idx]))
            if chosen_groups < min_chosen_groups or rem_groups < min_remaining_groups:
                continue

            ratio = len(chosen_idx) / n_total
            ratio_err = abs(ratio - target_ratio)

            # Strongly prefer keeping every class represented in the remaining (train) split
            rem_classes = set(np.unique(y[rem_idx]).tolist())
            missing_penalty = 10.0 * len(all_classes - rem_classes)

            # Mild penalty if chosen split misses classes
            chosen_classes = set(np.unique(y[chosen_idx]).tolist())
            chosen_missing_penalty = 0.25 * len(all_classes - chosen_classes)

            score = ratio_err + missing_penalty + chosen_missing_penalty
            if score < best_score:
                best_score = score
                best_chosen = combo

    if best_chosen is None:
        # Fallback: pick first fold
        best_chosen = (0,)

    chosen_idx = np.sort(np.concatenate([folds[i] for i in best_chosen]))
    rem_idx = np.sort(
        np.concatenate([folds[i] for i in range(n_folds) if i not in best_chosen])
    )
    return rem_idx, chosen_idx


def signer_aware_stratified_split(
    y: Sequence[int] | np.ndarray,
    signer_ids: Sequence[int | str] | np.ndarray,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition sample indices into train / val / test with ZERO signer overlap.

    Uses ``StratifiedGroupKFold`` grouped by ``signer_id`` so no signer ever
    appears in more than one split while preserving class stratification as
    closely as possible to ``(1 - val_split - test_split) / val_split / test_split``.
    """
    y_arr = np.asarray(y)
    groups = np.asarray(signer_ids)
    n_samples = len(y_arr)
    unique_signers = np.unique(groups)

    if len(unique_signers) < 3:
        raise ValueError(
            f"Signer-aware 3-way split requires at least 3 distinct signers; got {len(unique_signers)}."
        )

    temp_ratio = val_split + test_split
    if not (0.0 < temp_ratio < 1.0):
        raise ValueError(f"val_split + test_split must be in (0, 1); got {temp_ratio}")

    # Step 1: train vs (val + test)
    n_splits_outer = min(10, len(unique_signers))
    sgkf_outer = StratifiedGroupKFold(
        n_splits=n_splits_outer, shuffle=True, random_state=seed
    )
    outer_folds = [
        val_idx for _, val_idx in sgkf_outer.split(np.zeros(n_samples), y_arr, groups=groups)
    ]

    train_idx, temp_idx = _choose_fold_subset(
        outer_folds,
        y=y_arr,
        groups=groups,
        target_ratio=temp_ratio,
        min_remaining_groups=1,
        min_chosen_groups=2,
    )

    # Step 2: (val + test) -> val vs test
    temp_y = y_arr[temp_idx]
    temp_groups = groups[temp_idx]
    n_temp_signers = len(np.unique(temp_groups))
    n_splits_inner = min(6, n_temp_signers)

    sgkf_inner = StratifiedGroupKFold(
        n_splits=n_splits_inner, shuffle=True, random_state=seed
    )
    inner_folds = [
        fold_idx
        for _, fold_idx in sgkf_inner.split(np.zeros(len(temp_idx)), temp_y, groups=temp_groups)
    ]

    rel_test_ratio = test_split / temp_ratio
    val_rel, test_rel = _choose_fold_subset(
        inner_folds,
        y=temp_y,
        groups=temp_groups,
        target_ratio=rel_test_ratio,
        min_remaining_groups=1,
        min_chosen_groups=1,
    )

    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]

    # Enforce strict zero signer overlap across all splits
    train_signers = set(groups[train_idx].tolist())
    val_signers = set(groups[val_idx].tolist())
    test_signers = set(groups[test_idx].tolist())

    assert train_signers.isdisjoint(val_signers), "Signer leakage between train and val!"
    assert train_signers.isdisjoint(test_signers), "Signer leakage between train and test!"
    assert val_signers.isdisjoint(test_signers), "Signer leakage between val and test!"

    logger.info(
        "Signer-aware split: train=%d (%d signers) | val=%d (%d signers) | test=%d (%d signers)",
        len(train_idx), len(train_signers),
        len(val_idx), len(val_signers),
        len(test_idx), len(test_signers),
    )
    return train_idx, val_idx, test_idx


# ─── Class-Subfolder Fallback Layout ─────────────────────────────────────────

def discover_samples(
    feature_dir: str | Path,
) -> Tuple[List[Tuple[Path, int]], Dict[str, int], Dict[int, str]]:
    """Walk *feature_dir* (class-subfolder fallback layout) and collect .npy files."""
    feature_dir = Path(feature_dir)
    if not feature_dir.exists():
        raise FileNotFoundError(f"Feature directory not found: {feature_dir}")

    class_dirs = sorted(p for p in feature_dir.iterdir() if p.is_dir())
    if not class_dirs:
        raise ValueError(f"No class sub-directories found in {feature_dir}")

    label_to_idx = {d.name: i for i, d in enumerate(class_dirs)}
    idx_to_label = {i: d.name for i, d in enumerate(class_dirs)}

    samples: List[Tuple[Path, int]] = []
    for d in class_dirs:
        label = label_to_idx[d.name]
        for f in sorted(d.glob("*.npy")):
            samples.append((f, label))

    logger.info(
        "Discovered %d samples across %d classes in %s",
        len(samples), len(class_dirs), feature_dir,
    )
    return samples, label_to_idx, idx_to_label


# ─── DataLoader Factory ──────────────────────────────────────────────────────

def _make_weighted_sampler(train_labels: np.ndarray | List[int], num_classes: int) -> WeightedRandomSampler:
    class_counts = np.bincount(train_labels, minlength=num_classes)
    weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = torch.tensor(
        [weights[int(lbl)] for lbl in train_labels], dtype=torch.float
    )
    return WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )


def build_dataloaders(
    data_path: str | Path = "data/wlasl10_landmarks.npz",
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    wlasl_json_path: str | Path = "data/WLASL_v0.3.json",
    raw_videos_dir: str | Path = "data/raw_videos",
    num_classes: int = 10,
    sequence_length: int = 30,
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    signer_ids: Optional[np.ndarray] = None,
    label_to_idx: Optional[Dict[str, int]] = None,
    idx_to_label: Optional[Dict[int, str]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], Dict[int, str]]:
    """Build train / val / test DataLoaders from WLASL cache, WLASL JSON+videos,
    in-memory arrays, or class-subfolder fallback directory.
    """
    # Mode A: Explicit in-memory arrays (e.g. smoke test or unit tests)
    if X is not None and y is not None:
        n_cls = int(np.max(y)) + 1 if label_to_idx is None else len(label_to_idx)
        if label_to_idx is None:
            label_to_idx = {f"class_{i}": i for i in range(n_cls)}
        if idx_to_label is None:
            idx_to_label = {i: k for k, i in label_to_idx.items()}

        if signer_ids is not None:
            tr_idx, va_idx, te_idx = signer_aware_stratified_split(
                y, signer_ids, val_split=val_split, test_split=test_split, seed=seed
            )
        else:
            indices = np.arange(len(y))
            tr_idx, tmp_idx, _, tmp_y = train_test_split(
                indices, y, test_size=val_split + test_split, stratify=y, random_state=seed
            )
            rel_test = test_split / (val_split + test_split)
            va_idx, te_idx = train_test_split(
                tmp_idx, test_size=rel_test, stratify=tmp_y, random_state=seed
            )

        train_ds = SignLanguageDataset(X=X[tr_idx], y=y[tr_idx], label_to_idx=label_to_idx, augment=True, seed=seed)
        val_ds = SignLanguageDataset(X=X[va_idx], y=y[va_idx], label_to_idx=label_to_idx, augment=False)
        test_ds = SignLanguageDataset(X=X[te_idx], y=y[te_idx], label_to_idx=label_to_idx, augment=False)

        sampler = _make_weighted_sampler(y[tr_idx], len(label_to_idx))
        return (
            DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
            label_to_idx,
            idx_to_label,
        )

    data_path_p = Path(data_path)

    # Mode B: Cached WLASL .npz file (or build from WLASL_v0.3.json + raw_videos/)
    if data_path_p.suffix == ".npz":
        bundle = prepare_wlasl_dataset(
            wlasl_json_path=wlasl_json_path,
            raw_videos_dir=raw_videos_dir,
            cache_path=data_path_p,
            num_classes=num_classes,
            sequence_length=sequence_length,
        )
        return build_dataloaders(
            batch_size=batch_size,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
            X=bundle["X"],                # type: ignore[arg-type]
            y=bundle["y"],                # type: ignore[arg-type]
            signer_ids=bundle["signer_id"],  # type: ignore[arg-type]
            label_to_idx=bundle["label_to_idx"],  # type: ignore[arg-type]
            idx_to_label=bundle["idx_to_label"],  # type: ignore[arg-type]
        )

    # Mode C: Directory path — check if WLASL flat videos exist, else class-subfolder fallback
    if data_path_p.is_dir():
        has_subdirs = any(p.is_dir() for p in data_path_p.iterdir())
        if has_subdirs:
            samples, label_to_idx, idx_to_label = discover_samples(data_path_p)
            labels = [s[1] for s in samples]

            train_samp, valtest_samp, _, valtest_labels = train_test_split(
                samples, labels,
                test_size=val_split + test_split,
                stratify=labels,
                random_state=seed,
            )
            relative_test = test_split / (val_split + test_split)
            val_samp, test_samp = train_test_split(
                valtest_samp,
                test_size=relative_test,
                stratify=valtest_labels,
                random_state=seed,
            )

            train_labels = [s[1] for s in train_samp]
            sampler = _make_weighted_sampler(train_labels, len(label_to_idx))

            train_ds = SignLanguageDataset(samples=train_samp, label_to_idx=label_to_idx, augment=True, seed=seed)
            val_ds = SignLanguageDataset(samples=val_samp, label_to_idx=label_to_idx, augment=False)
            test_ds = SignLanguageDataset(samples=test_samp, label_to_idx=label_to_idx, augment=False)

            return (
                DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory),
                DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
                DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
                label_to_idx,
                idx_to_label,
            )
        else:
            # Flat raw_videos directory with WLASL JSON
            cache_file = data_path_p.parent / "wlasl10_landmarks.npz"
            bundle = prepare_wlasl_dataset(
                wlasl_json_path=wlasl_json_path,
                raw_videos_dir=data_path_p,
                cache_path=cache_file,
                num_classes=num_classes,
                sequence_length=sequence_length,
            )
            return build_dataloaders(
                batch_size=batch_size,
                val_split=val_split,
                test_split=test_split,
                seed=seed,
                num_workers=num_workers,
                pin_memory=pin_memory,
                X=bundle["X"],                # type: ignore[arg-type]
                y=bundle["y"],                # type: ignore[arg-type]
                signer_ids=bundle["signer_id"],  # type: ignore[arg-type]
                label_to_idx=bundle["label_to_idx"],  # type: ignore[arg-type]
                idx_to_label=bundle["idx_to_label"],  # type: ignore[arg-type]
            )

    # Path doesn't exist yet — if WLASL metadata exists, prepare & cache
    if Path(wlasl_json_path).is_file() and Path(raw_videos_dir).is_dir():
        bundle = prepare_wlasl_dataset(
            wlasl_json_path=wlasl_json_path,
            raw_videos_dir=raw_videos_dir,
            cache_path=data_path_p,
            num_classes=num_classes,
            sequence_length=sequence_length,
        )
        return build_dataloaders(
            batch_size=batch_size,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
            X=bundle["X"],                # type: ignore[arg-type]
            y=bundle["y"],                # type: ignore[arg-type]
            signer_ids=bundle["signer_id"],  # type: ignore[arg-type]
            label_to_idx=bundle["label_to_idx"],  # type: ignore[arg-type]
            idx_to_label=bundle["idx_to_label"],  # type: ignore[arg-type]
        )

    raise FileNotFoundError(f"Data source not found: {data_path}")


# ─── Isolated Smoke-Test Generator ───────────────────────────────────────────

def generate_smoke_test_data(
    num_classes: int = 10,
    samples_per_class: int = 12,
    num_signers: int = 8,
    sequence_length: int = 30,
    num_features: int = N_FEATURES,
    seed: int = 42,
) -> Dict[str, object]:
    """Generate temporary in-memory tensors strictly for unit/smoke testing.

    Completely isolated from ``data/wlasl10_landmarks.npz`` (never reads or writes disk).
    """
    rng = np.random.default_rng(seed)
    n_total = num_classes * samples_per_class

    coords = rng.normal(0.0, 0.25, size=(n_total, sequence_length, num_features // 2)).astype(np.float32)
    vel = np.zeros_like(coords, dtype=np.float32)
    vel[:, 1:, :] = coords[:, 1:, :] - coords[:, :-1, :]
    X = np.concatenate([coords, vel], axis=-1).astype(np.float32)

    y = np.repeat(np.arange(num_classes, dtype=np.int64), samples_per_class)
    # Assign signer_ids across samples so every signer has multiple classes
    signer_ids = np.asarray(
        [(i + c * 3) % num_signers for c in range(num_classes) for i in range(samples_per_class)],
        dtype=np.int64,
    )
    classes = [f"smoke_gloss_{i:02d}" for i in range(num_classes)]
    gloss = np.asarray([classes[lbl] for lbl in y])
    label_to_idx = {name: i for i, name in enumerate(classes)}
    idx_to_label = {i: name for i, name in enumerate(classes)}

    return {
        "X": X,
        "y": y,
        "gloss": gloss,
        "signer_id": signer_ids,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
    }


def save_sample_npy_files(
    bundle: Dict[str, object],
    samples_dir: str | Path = "data/samples",
    num_samples: int = 3,
    seed: int = 42,
) -> Path:
    """Save lightweight (30, 450) .npy test samples + ground-truth labels for reviewers."""
    samples_path = Path(samples_dir)
    samples_path.mkdir(parents=True, exist_ok=True)

    X = bundle["X"]  # type: ignore[index]
    y = bundle["y"]  # type: ignore[index]
    gloss = bundle["gloss"]  # type: ignore[index]
    signer_id = bundle["signer_id"]  # type: ignore[index]
    video_id = bundle["video_id"]  # type: ignore[index]

    _, _, test_idx = signer_aware_stratified_split(y, signer_id, seed=seed)

    # Pick up to *num_samples* test samples with distinct glosses
    chosen_indices: List[int] = []
    seen_glosses = set()
    for idx in test_idx:
        g = str(gloss[idx])
        if g not in seen_glosses:
            seen_glosses.add(g)
            chosen_indices.append(int(idx))
        if len(chosen_indices) >= num_samples:
            break
    for idx in test_idx:
        if len(chosen_indices) >= num_samples:
            break
        if int(idx) not in chosen_indices:
            chosen_indices.append(int(idx))

    manifest = []
    for k, idx in enumerate(chosen_indices, start=1):
        g = str(gloss[idx])
        fname = f"sample_{k}_{g}.npy"
        fpath = samples_path / fname
        np.save(fpath, np.asarray(X[idx], dtype=np.float32))
        manifest.append({
            "file": fname,
            "ground_truth_gloss": g,
            "class_idx": int(y[idx]),
            "signer_id": int(signer_id[idx]),
            "video_id": str(video_id[idx]),
            "shape": list(X[idx].shape),
        })

    manifest_path = samples_path / "samples_labels.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved %d sample .npy files → %s", len(manifest), samples_path)
    return manifest_path


# ─── CLI entry point ─────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WLASL Dataset Preparation & Smoke Testing")
    p.add_argument("--wlasl-json", default="data/WLASL_v0.3.json", help="Path to WLASL_v0.3.json")
    p.add_argument("--raw-videos", default="data/raw_videos", help="Directory containing <video_id>.mp4")
    p.add_argument("--cache", default="data/wlasl10_landmarks.npz", help="Output .npz cache path")
    p.add_argument("--samples-dir", default="data/samples", help="Directory for 3 lightweight reviewer .npy samples")
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--sequence-length", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true", help="Re-extract even if cache exists")
    p.add_argument(
        "--smoke-test-data",
        action="store_true",
        help="Generate temporary in-memory tensors only (isolated from data/wlasl10_landmarks.npz)",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    args = _parse_args()

    if args.smoke_test_data:
        bundle = generate_smoke_test_data(
            num_classes=args.num_classes,
            sequence_length=args.sequence_length,
            seed=args.seed,
        )
        tr_idx, va_idx, te_idx = signer_aware_stratified_split(
            bundle["y"], bundle["signer_id"], seed=args.seed  # type: ignore[arg-type]
        )
        train_loader, val_loader, test_loader, _, _ = build_dataloaders(
            X=bundle["X"],                  # type: ignore[arg-type]
            y=bundle["y"],                  # type: ignore[arg-type]
            signer_ids=bundle["signer_id"],  # type: ignore[arg-type]
            label_to_idx=bundle["label_to_idx"],  # type: ignore[arg-type]
            idx_to_label=bundle["idx_to_label"],  # type: ignore[arg-type]
            seed=args.seed,
        )
        xb, yb = next(iter(train_loader))
        logger.info(
            "Smoke-test complete (in-memory only): X=%s  splits=(%d, %d, %d)  batch=%s/%s",
            bundle["X"].shape, len(tr_idx), len(va_idx), len(te_idx), xb.shape, yb.shape,  # type: ignore[union-attr]
        )
        return

    bundle = prepare_wlasl_dataset(
        wlasl_json_path=args.wlasl_json,
        raw_videos_dir=args.raw_videos,
        cache_path=args.cache,
        num_classes=args.num_classes,
        sequence_length=args.sequence_length,
        force_recompute=args.force,
    )
    save_sample_npy_files(bundle, samples_dir=args.samples_dir, num_samples=3, seed=args.seed)


if __name__ == "__main__":
    main()
