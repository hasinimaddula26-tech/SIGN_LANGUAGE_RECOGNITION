"""
tests/test_pipeline.py
──────────────────────
Comprehensive pytest suite for LandmarkExtractor, WLASL dataset preparation,
signer-aware splitting, kinematic augmentation, all 4 models, and integration.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

# Allow ``import src.xxx`` when running from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import pytest
import torch

from src.dataset import (
    SignLanguageDataset,
    build_dataloaders,
    generate_smoke_test_data,
    load_wlasl_cache,
    prepare_wlasl_dataset,
    select_top_wlasl_glosses,
    signer_aware_stratified_split,
)
from src.features import (
    LandmarkExtractor,
    N_COORDS,
    N_FEATURES,
    _compute_velocity,
    _normalise_frame,
    _resample,
    trim_active_motion,
)
from src.models import (
    CNNBaseline,
    CNNLSTMStandard,
    LSTMBaseline,
    ProposedResCNNBiLSTMAttn,
    build_model,
)

# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="class")
def extractor():
    """Per-class LandmarkExtractor so MediaPipe memory is released between classes."""
    import gc
    ext = LandmarkExtractor(sequence_length=30, static_image_mode=True)
    yield ext
    ext.close()
    gc.collect()


def make_frames(n: int, h: int = 128, w: int = 128) -> list:
    """Create *n* synthetic BGR uint8 frames (random noise)."""
    rng = np.random.default_rng(42)
    return [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(n)]


def write_dummy_mp4(path, n_frames: int = 15, h: int = 64, w: int = 64) -> None:
    """Write a short synthetic MP4 video for WLASL pipeline tests."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 25.0, (w, h))
    rng = np.random.default_rng(123)
    for _ in range(n_frames):
        frame = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


# ─── Shape & dtype tests ─────────────────────────────────────────────────────

class TestOutputShape:

    def test_nominal_30_frames(self, extractor):
        frames = make_frames(30)
        out = extractor.process_frames(frames)
        assert out.shape == (30, 450), f"Expected (30, 450), got {out.shape}"

    def test_short_12_frames(self, extractor):
        frames = make_frames(12)
        out = extractor.process_frames(frames)
        assert out.shape == (30, 450), f"Short clip: expected (30, 450), got {out.shape}"

    def test_long_75_frames(self, extractor):
        frames = make_frames(75)
        out = extractor.process_frames(frames)
        assert out.shape == (30, 450), f"Long clip: expected (30, 450), got {out.shape}"

    def test_dtype_float32(self, extractor):
        frames = make_frames(30)
        out = extractor.process_frames(frames)
        assert out.dtype == np.float32, f"Expected float32, got {out.dtype}"

    def test_no_nan(self, extractor):
        frames = make_frames(30)
        out = extractor.process_frames(frames)
        assert not np.isnan(out).any(), "NaN found in output!"

    def test_no_inf(self, extractor):
        frames = make_frames(30)
        out = extractor.process_frames(frames)
        assert not np.isinf(out).any(), "Inf found in output!"


# ─── Edge case: zero / empty frames ──────────────────────────────────────────

class TestEdgeCases:

    def test_zero_frames_returns_zeros(self, extractor):
        out = extractor.process_frames([])
        assert out.shape == (30, 450)
        assert not np.isnan(out).any()

    def test_all_black_frames(self, extractor):
        """All-zero frames → no landmarks detected → safe zeros."""
        frames = [np.zeros((128, 128, 3), dtype=np.uint8) for _ in range(30)]
        out = extractor.process_frames(frames)
        assert out.shape == (30, 450)
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()

    def test_single_frame(self, extractor):
        """Single frame should be edge-padded to 30."""
        frames = make_frames(1)
        out = extractor.process_frames(frames)
        assert out.shape == (30, 450)
        assert not np.isnan(out).any()


# ─── Velocity semantics ───────────────────────────────────────────────────────

class TestVelocity:

    def _get_coords_vel(self, extractor, n_frames: int = 10):
        frames = make_frames(n_frames)
        out = extractor.process_frames(frames)   # (30, 450)
        coords = out[:, :N_COORDS]               # (30, 225)
        vel    = out[:, N_COORDS:]               # (30, 225)
        return coords, vel

    def test_first_frame_velocity_is_zero(self, extractor):
        _, vel = self._get_coords_vel(extractor, 30)
        np.testing.assert_array_equal(
            vel[0], np.zeros(N_COORDS, dtype=np.float32),
            err_msg="Velocity at t=0 must be zero."
        )

    def test_velocity_equals_finite_difference(self, extractor):
        """vel[t] == coords[t] - coords[t-1] for t >= 1."""
        frames = make_frames(30)
        out = extractor.process_frames(frames)
        coords = out[:, :N_COORDS]
        vel    = out[:, N_COORDS:]
        expected = np.diff(coords, axis=0)       # (29, 225)
        np.testing.assert_allclose(
            vel[1:], expected, atol=1e-5,
            err_msg="Velocity does not match finite difference."
        )


# ─── Resample unit tests ──────────────────────────────────────────────────────

class TestResample:

    def test_exact_length_unchanged(self):
        arr = np.random.randn(30, 450).astype(np.float32)
        out = _resample(arr, 30)
        np.testing.assert_array_equal(arr, out)

    def test_short_padded_to_target(self):
        arr = np.random.randn(12, 450).astype(np.float32)
        out = _resample(arr, 30)
        assert out.shape == (30, 450)
        np.testing.assert_array_equal(out[12:], np.repeat(arr[[-1]], 18, axis=0))

    def test_long_subsampled_to_target(self):
        arr = np.random.randn(75, 450).astype(np.float32)
        out = _resample(arr, 30)
        assert out.shape == (30, 450)

    def test_trim_active_motion_removes_idle_ends(self):
        arr = np.zeros((40, 225), dtype=np.float32)
        # Active gesture in frames 12..27 across hand coordinates 99:225
        for t in range(12, 28):
            arr[t, 99:225] = float(t - 11) * 0.1
        # Keep trailing pose/hands static at last active value
        arr[28:, 99:225] = arr[27, 99:225]
        trimmed = trim_active_motion(arr, margin=3)
        assert 15 <= len(trimmed) < 40
        assert len(trimmed) == (27 - 12 + 1) + 6  # 16 active frames + 3 margin each side

    def test_trim_active_motion_fallback_when_few_active_frames(self):
        arr = np.zeros((25, 225), dtype=np.float32)
        # Only 3 frames with movement (< 8 active frames) -> fallback to full clip
        arr[10, 99:225] = 0.5
        arr[11, 99:225] = 1.0
        arr[12, 99:225] = 0.2
        trimmed = trim_active_motion(arr, margin=3)
        assert trimmed.shape == arr.shape


# ─── Normalisation unit tests ─────────────────────────────────────────────────

class TestNormalisation:

    def _make_keypoints(self):
        """75 keypoints with known shoulder positions."""
        kp = np.zeros((75, 3), dtype=np.float32)
        kp[11] = [0.4, 0.3, 0.0]   # left shoulder  (pose idx 11)
        kp[12] = [0.6, 0.3, 0.0]   # right shoulder (pose idx 12)
        kp[33] = [0.5, 0.5, 0.1]   # left hand wrist
        kp[54] = [0.5, 0.2, 0.1]   # right hand wrist
        return kp

    def test_mid_shoulder_becomes_origin(self):
        kp = self._make_keypoints()
        normed = _normalise_frame(kp)
        mid = (normed[11] + normed[12]) / 2.0
        np.testing.assert_allclose(
            mid, [0.0, 0.0, 0.0], atol=1e-5,
            err_msg="Normalised mid-shoulder should be at origin."
        )

    def test_shoulder_width_becomes_one(self):
        kp = self._make_keypoints()
        normed = _normalise_frame(kp)
        width = float(np.linalg.norm(normed[11] - normed[12]))
        assert abs(width - 1.0) < 1e-5, (
            f"Normalised shoulder width should be 1.0, got {width:.6f}"
        )

    def test_fallback_on_zero_pose(self):
        """All-zero pose (no detection) → no crash, no NaN."""
        kp = np.zeros((75, 3), dtype=np.float32)
        normed = _normalise_frame(kp)
        assert not np.isnan(normed).any()
        assert not np.isinf(normed).any()

    def test_hands_normalised_relative_to_body(self):
        """Hand landmarks should NOT be centered on wrist; body-offset preserved."""
        kp = self._make_keypoints()
        normed = _normalise_frame(kp)
        expected = np.array([0.0, 1.0, 0.5], dtype=np.float32)
        np.testing.assert_allclose(normed[33], expected, atol=1e-5)

    def test_undetected_hand_stays_zero(self):
        """Undetected hand block remains zero even when pose is detected."""
        kp = self._make_keypoints()
        kp[54:75] = 0.0
        normed = _normalise_frame(kp)
        np.testing.assert_array_equal(normed[54:75], np.zeros((21, 3), dtype=np.float32))
        assert np.any(normed[33:54] != 0.0)


# ─── Missing-landmark robustness ──────────────────────────────────────────────

class TestMissingLandmarks:

    def test_missing_hands_all_black(self, extractor):
        frames = [np.zeros((128, 128, 3), dtype=np.uint8)] * 30
        out = extractor.process_frames(frames)
        assert out.shape == (30, 450)
        assert not np.isnan(out).any()

    def test_mixed_frames_no_crash(self, extractor):
        frames = []
        rng = np.random.default_rng(7)
        for i in range(30):
            if i % 3 == 0:
                frames.append(np.zeros((64, 64, 3), dtype=np.uint8))
            else:
                frames.append(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
        out = extractor.process_frames(frames)
        assert out.shape == (30, 450)
        assert not np.isnan(out).any()


# ─── Compute velocity unit test ───────────────────────────────────────────────

class TestComputeVelocity:

    def test_zero_first_row(self):
        coords = np.ones((5, 10), dtype=np.float32)
        vel = _compute_velocity(coords)
        np.testing.assert_array_equal(vel[0], np.zeros(10, dtype=np.float32))

    def test_correct_differences(self):
        coords = np.arange(20, dtype=np.float32).reshape(4, 5)
        vel = _compute_velocity(coords)
        expected = np.vstack([np.zeros((1, 5)), np.diff(coords, axis=0)])
        np.testing.assert_allclose(vel, expected.astype(np.float32))


# ─── WLASL Dataset, Signer-Aware Splitting & Augmentation Tests ──────────────

class TestDatasetWLASL:

    def test_wlasl_top10_selection_caching_and_frame_slicing(self, tmp_path):
        raw_dir = tmp_path / "raw_videos"
        raw_dir.mkdir()
        json_path = tmp_path / "WLASL_v0.3.json"
        cache_path = tmp_path / "wlasl10_landmarks.npz"

        # Create 12 glosses; top 10 have 2 available videos, bottom 2 have 1 video
        entries = []
        vid_counter = 1000
        for g_idx in range(12):
            gloss_name = f"gloss_{g_idx:02d}"
            n_avail = 2 if g_idx < 10 else 1
            instances = []
            for j in range(n_avail):
                vid_id = str(vid_counter)
                vid_counter += 1
                write_dummy_mp4(raw_dir / f"{vid_id}.mp4", n_frames=12)
                instances.append({
                    "video_id": vid_id,
                    "signer_id": (g_idx * 3 + j) % 8,
                    "split": "train",
                    "frame_start": 2 if j == 0 else 1,
                    "frame_end": 10 if j == 0 else -1,
                })
            # Also add a missing video_id entry that must be ignored
            instances.append({
                "video_id": f"missing_{g_idx}",
                "signer_id": 99,
                "split": "val",
                "frame_start": 1,
                "frame_end": -1,
            })
            entries.append({"gloss": gloss_name, "instances": instances})

        json_path.write_text(json.dumps(entries), encoding="utf-8")

        selected = select_top_wlasl_glosses(json_path, raw_dir, num_classes=10)
        assert len(selected) == 10
        assert all(len(insts) == 2 for _, insts in selected)

        bundle = prepare_wlasl_dataset(
            wlasl_json_path=json_path,
            raw_videos_dir=raw_dir,
            cache_path=cache_path,
            num_classes=10,
            sequence_length=30,
        )
        assert cache_path.is_file()
        assert bundle["X"].shape == (20, 30, 450)
        assert bundle["y"].shape == (20,)
        assert len(bundle["label_to_idx"]) == 10

        reloaded = load_wlasl_cache(cache_path)
        np.testing.assert_array_equal(bundle["X"], reloaded["X"])
        np.testing.assert_array_equal(bundle["y"], reloaded["y"])
        np.testing.assert_array_equal(bundle["signer_id"], reloaded["signer_id"])

    def test_class_subfolder_fallback_layout(self, tmp_path):
        feat_dir = tmp_path / "features"
        for cls_name in ["hello", "thanks", "yes"]:
            cls_dir = feat_dir / cls_name
            cls_dir.mkdir(parents=True)
            for i in range(10):
                arr = np.random.randn(30, 450).astype(np.float32)
                np.save(cls_dir / f"sample_{i}.npy", arr)

        tr_loader, va_loader, te_loader, label_to_idx, idx_to_label = build_dataloaders(
            feat_dir, batch_size=4, val_split=0.2, test_split=0.2, seed=42
        )
        assert len(label_to_idx) == 3
        assert len(idx_to_label) == 3
        xb, yb = next(iter(tr_loader))
        assert xb.shape[1:] == (30, 450)
        assert len(va_loader.dataset) > 0
        assert len(te_loader.dataset) > 0

    def test_signer_aware_stratified_split_zero_overlap(self):
        smoke = generate_smoke_test_data(
            num_classes=10, samples_per_class=16, num_signers=10, seed=42
        )
        y = smoke["y"]
        signers = smoke["signer_id"]

        tr_idx, va_idx, te_idx = signer_aware_stratified_split(
            y, signers, val_split=0.15, test_split=0.15, seed=42
        )

        tr_signers = set(signers[tr_idx].tolist())
        va_signers = set(signers[va_idx].tolist())
        te_signers = set(signers[te_idx].tolist())

        assert tr_signers.isdisjoint(va_signers), "Train/val signer overlap!"
        assert tr_signers.isdisjoint(te_signers), "Train/test signer overlap!"
        assert va_signers.isdisjoint(te_signers), "Val/test signer overlap!"
        assert len(tr_idx) + len(va_idx) + len(te_idx) == len(y)
        assert len(np.unique(y[tr_idx])) == 10

    def test_augmentation_strictly_on_train_only(self):
        smoke = generate_smoke_test_data(
            num_classes=10, samples_per_class=12, num_signers=8, seed=42
        )
        tr_loader, va_loader, te_loader, _, _ = build_dataloaders(
            X=smoke["X"],
            y=smoke["y"],
            signer_ids=smoke["signer_id"],
            label_to_idx=smoke["label_to_idx"],
            idx_to_label=smoke["idx_to_label"],
            batch_size=8,
            seed=42,
        )

        assert tr_loader.dataset.augment is True
        assert va_loader.dataset.augment is False
        assert te_loader.dataset.augment is False

        # Val & Test datasets must return deterministic, un-augmented tensors
        vx1, _ = va_loader.dataset[0]
        vx2, _ = va_loader.dataset[0]
        assert torch.equal(vx1, vx2)

        tx1, _ = te_loader.dataset[0]
        tx2, _ = te_loader.dataset[0]
        assert torch.equal(tx1, tx2)

        # Train dataset applies speed perturbation + Gaussian jitter -> differs across calls
        rx1, _ = tr_loader.dataset[0]
        rx2, _ = tr_loader.dataset[0]
        assert rx1.shape == (30, 450)
        assert not torch.allclose(rx1, rx2)
        assert not torch.isnan(rx1).any()
        # Velocity at t=0 remains 0 after augmentation
        assert torch.allclose(rx1[0, N_COORDS:], torch.zeros(N_COORDS))

    def test_smoke_test_cli_isolated(self, tmp_path):
        """--smoke-test-data CLI runs in memory without creating data/wlasl10_landmarks.npz."""
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cache_file = os.path.join(repo_root, "data", "wlasl10_landmarks.npz")
        existed_before = os.path.exists(cache_file)

        proc = subprocess.run(
            [sys.executable, "-m", "src.dataset", "--smoke-test-data"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "Smoke-test complete" in proc.stderr or "Smoke-test complete" in proc.stdout
        if not existed_before:
            assert not os.path.exists(cache_file)


# ─── Model forward pass (all 4 models + factory) ─────────────────────────────

class TestModels:

    @pytest.mark.parametrize(
        "model_name,cls",
        [
            ("CNNBaseline", CNNBaseline),
            ("LSTMBaseline", LSTMBaseline),
            ("CNNLSTMStandard", CNNLSTMStandard),
        ],
    )
    def test_standard_models_forward(self, model_name, cls):
        model = build_model(model_name, num_features=450, num_classes=10)
        assert isinstance(model, cls)
        assert model.count_parameters() > 0

        x = torch.randn(4, 30, 450)
        logits = model(x)
        assert logits.shape == (4, 10)
        assert not torch.isnan(logits).any()

        model.eval()
        with torch.no_grad():
            single = model(torch.randn(1, 30, 450))
        assert single.shape == (1, 10)
        assert not torch.isnan(single).any()

    def test_proposed_res_cnn_bilstm_attn_forward(self):
        model = build_model("ProposedResCNNBiLSTMAttn", num_features=450, num_classes=10)
        assert isinstance(model, ProposedResCNNBiLSTMAttn)
        assert model.count_parameters() > 0

        x = torch.randn(4, 30, 450)
        out = model(x)
        assert isinstance(out, tuple) and len(out) == 2
        logits, attn = out

        assert logits.shape == (4, 10)
        assert attn.shape == (4, 30)
        assert not torch.isnan(logits).any()
        assert not torch.isnan(attn).any()
        assert torch.all(attn >= 0.0)
        torch.testing.assert_close(attn.sum(dim=-1), torch.ones(4), atol=1e-5, rtol=1e-5)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            build_model("NonExistentModel")


# ─── Integration: LandmarkExtractor → ProposedResCNNBiLSTMAttn ────────────────

class TestIntegration:

    def test_extractor_to_proposed_model(self, extractor):
        frames = make_frames(30)
        feat = extractor.process_frames(frames)
        assert feat.shape == (30, 450)
        assert not np.isnan(feat).any()

        model = build_model("ProposedResCNNBiLSTMAttn", num_features=450, num_classes=10)
        model.eval()
        x = torch.from_numpy(feat).unsqueeze(0)
        with torch.no_grad():
            logits, attn = model(x)
        assert logits.shape == (1, 10)
        assert attn.shape == (1, 30)
        torch.testing.assert_close(attn.sum(dim=-1), torch.ones(1), atol=1e-5, rtol=1e-5)
        assert not torch.isnan(logits).any()

    @pytest.mark.parametrize("model_name", ["CNNBaseline", "ProposedResCNNBiLSTMAttn"])
    def test_train_eval_predict_end_to_end(self, tmp_path, model_name):
        """Train 1 epoch → checkpoint → load_model → Evaluator → Predictor."""
        import gc
        from src.evaluate import Evaluator
        from src.models import load_model
        from src.predict import Predictor
        from src.train import Trainer

        smoke = generate_smoke_test_data(
            num_classes=10, samples_per_class=6, num_signers=6, seed=42
        )
        tr_loader, va_loader, te_loader, label_to_idx, idx_to_label = build_dataloaders(
            X=smoke["X"],
            y=smoke["y"],
            signer_ids=smoke["signer_id"],
            label_to_idx=smoke["label_to_idx"],
            idx_to_label=smoke["idx_to_label"],
            batch_size=8,
            seed=42,
        )

        cfg = {
            "model_name": model_name,
            "num_features": 450,
            "num_classes": 10,
            "cnn_out_channels": [32, 64],
            "cnn_kernel_size": 3,
            "cnn_dropout": 0.1,
            "lstm_hidden_size": 32,
            "lstm_num_layers": 2,
            "lstm_dropout": 0.1,
            "fc_dropout": 0.1,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "num_epochs": 1,
            "early_stopping_patience": 2,
            "grad_clip": 1.0,
            "seed": 42,
        }

        out_dir = tmp_path / "models"
        trainer = Trainer(
            cfg, tr_loader, va_loader,
            device="cpu", out_dir=out_dir,
            idx_to_label=idx_to_label, label_to_idx=label_to_idx,
        )
        history = trainer.fit()
        assert len(history["train_loss"]) == 1
        ckpt_path = out_dir / "best_model.pt"
        assert ckpt_path.is_file()
        assert (out_dir / "history.json").is_file()

        loaded = load_model(str(ckpt_path), cfg, device="cpu")
        evaluator = Evaluator(loaded, te_loader, idx_to_label, device="cpu")
        report = evaluator.run()
        assert 0.0 <= report["accuracy"] <= 1.0
        assert 0.0 <= report["macro_f1"] <= 1.0
        assert len(report["confusion_matrix"]) == 10

        cm_plot = tmp_path / "logs" / "cm.png"
        evaluator.save_confusion_matrix_plot(report, cm_plot)
        assert cm_plot.is_file()

        predictor = Predictor(loaded, None, idx_to_label, device="cpu")
        npy_sample = tmp_path / "sample.npy"
        np.save(npy_sample, smoke["X"][0])
        npy_label, npy_conf, npy_probs = predictor.predict_npy(str(npy_sample))
        assert npy_label in idx_to_label.values()
        assert 0.0 <= npy_conf <= 1.0
        assert npy_probs.shape == (10,)
        if model_name == "ProposedResCNNBiLSTMAttn":
            assert predictor.last_attention is not None
            assert predictor.last_attention.shape == (30,)
            assert np.isclose(predictor.last_attention.sum(), 1.0, atol=1e-5)

        del trainer, loaded, evaluator, predictor
        gc.collect()

    def test_train_all_and_benchmark_all_pipeline(self, tmp_path):
        """train_all_models + benchmark_all outputs valid CSVs, hardware specs, latencies, and 4 PNGs."""
        import csv
        from src.evaluate import COMPARISON_COLUMNS, benchmark_all
        from src.train import ALL_MODEL_NAMES, train_all_models

        smoke = generate_smoke_test_data(
            num_classes=10, samples_per_class=6, num_signers=6, seed=42
        )
        tr_loader, va_loader, te_loader, label_to_idx, idx_to_label = build_dataloaders(
            X=smoke["X"],
            y=smoke["y"],
            signer_ids=smoke["signer_id"],
            label_to_idx=smoke["label_to_idx"],
            idx_to_label=smoke["idx_to_label"],
            batch_size=16,
            seed=42,
        )

        cfg = {
            "num_features": 450,
            "num_classes": 10,
            "sequence_length": 30,
            "cnn_out_channels": [32, 64],
            "cnn_kernel_size": 3,
            "cnn_dropout": 0.1,
            "lstm_hidden_size": 32,
            "lstm_num_layers": 2,
            "lstm_dropout": 0.1,
            "fc_dropout": 0.1,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "num_epochs": 2,
            "early_stopping_patience": 2,
            "grad_clip": 1.0,
            "seed": 42,
        }

        ckpt_dir = tmp_path / "checkpoints"
        res_dir = tmp_path / "results"

        histories = train_all_models(
            config=cfg,
            train_loader=tr_loader,
            val_loader=va_loader,
            device="cpu",
            checkpoints_dir=ckpt_dir,
            results_dir=res_dir,
            idx_to_label=idx_to_label,
            label_to_idx=label_to_idx,
        )
        assert set(histories.keys()) == set(ALL_MODEL_NAMES)
        assert (res_dir / "all_models_history.json").is_file()
        for name in ALL_MODEL_NAMES:
            assert (ckpt_dir / f"{name}_best.pt").is_file()

        summary = benchmark_all(
            config=cfg,
            test_loader=te_loader,
            idx_to_label=idx_to_label,
            checkpoints_dir=ckpt_dir,
            results_dir=res_dir,
            device="cpu",
        )
        assert len(summary["comparison_rows"]) == 4

        # Verify model_comparison.csv exact columns & non-zero latency
        comp_csv = res_dir / "model_comparison.csv"
        assert comp_csv.is_file()
        with open(comp_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == COMPARISON_COLUMNS
            rows = list(reader)
        assert len(rows) == 4
        for row in rows:
            assert float(row["CPU Latency Mean (ms)"]) > 0.0
            assert float(row["CPU Latency Std (ms)"]) >= 0.0
            assert int(row["Parameter Count"]) > 0
            assert float(row["Checkpoint Size (KB)"]) > 0.0

        # Verify hardware_specs.json & per_class_metrics.csv
        hw_json = res_dir / "hardware_specs.json"
        assert hw_json.is_file()
        hw = json.loads(hw_json.read_text(encoding="utf-8"))
        for key in ("os", "cpu", "python_version", "pytorch_version"):
            assert key in hw and hw[key]

        per_cls_csv = res_dir / "per_class_metrics.csv"
        assert per_cls_csv.is_file()
        with open(per_cls_csv, newline="", encoding="utf-8") as f:
            cls_rows = list(csv.DictReader(f))
        assert len(cls_rows) == 10

        # Verify all 4 300-DPI PNG charts exist and are non-empty
        for png_name in (
            "benchmark_comparison.png",
            "training_curves.png",
            "confusion_matrix_proposed.png",
            "temporal_attention_weights.png",
        ):
            png_path = res_dir / png_name
            assert png_path.is_file(), f"Missing {png_name}"
            assert png_path.stat().st_size > 0, f"Empty {png_name}"
