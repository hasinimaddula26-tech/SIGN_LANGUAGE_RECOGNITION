import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import app
from src.dataset import load_wlasl_cache

TARGET_GLOSSES = [
    "before", "computer", "cool", "cousin", "drink",
    "go", "help", "tall", "thin", "who",
]


def main() -> None:
    raw_dir = Path("data/raw_videos")
    demo_dir = Path("demo_videos")
    if demo_dir.exists():
        shutil.rmtree(demo_dir)
    demo_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Score all cached (30, 450) clips to rank top candidates per gloss
    bundle = load_wlasl_cache("data/wlasl10_landmarks.npz")
    X = bundle["X"]
    gloss_arr = bundle["gloss"]
    vid_arr = bundle["video_id"]

    candidates_by_gloss = defaultdict(list)
    for i in range(len(X)):
        true_g = str(gloss_arr[i])
        vid_id = str(vid_arr[i])
        mp4_path = raw_dir / f"{vid_id}.mp4"
        if not mp4_path.is_file():
            continue
        pred_g, conf, probs, _ = app._run_model_forward(X[i])
        true_idx = next(k for k, v in app.IDX_TO_LABEL.items() if v == true_g)
        true_prob = float(probs[true_idx])
        candidates_by_gloss[true_g].append({
            "video_id": vid_id,
            "mp4_path": mp4_path,
            "true_gloss": true_g,
            "cached_pred": pred_g,
            "cached_conf": conf,
            "cached_true_prob": true_prob,
        })

    for g in TARGET_GLOSSES:
        candidates_by_gloss[g].sort(
            key=lambda r: (r["cached_pred"] == g, r["cached_true_prob"], r["cached_conf"]),
            reverse=True,
        )

    # Stage 2: Verify candidates through exact app.py Tab 2 pipeline (predict_video_file)
    copied_rows = []
    verified_counts = {}

    for g in TARGET_GLOSSES:
        hits = []
        tested = 0
        for cand in candidates_by_gloss[g]:
            if len(hits) >= 2:
                break
            if tested >= 6 and len(hits) >= 1:
                break
            tested += 1
            _, probs_dict, _ = app.predict_video_file(str(cand["mp4_path"]), flip_horizontal=False)
            ranked = sorted(probs_dict.items(), key=lambda x: -x[1])
            pred_g, pred_conf = ranked[0]
            if pred_g == g:
                hits.append((cand["video_id"], cand["mp4_path"], g, pred_g, pred_conf))
                print(f"  ✓ MATCH [{g}] {cand['video_id']}.mp4 -> {pred_g} ({pred_conf*100:.2f}%)", flush=True)
            else:
                print(f"  · MISS  [{g}] {cand['video_id']}.mp4 -> {pred_g} ({pred_conf*100:.2f}%)", flush=True)

        verified_counts[g] = len(hits)
        for vid_id, mp4_path, true_g, pred_g, pred_conf in hits:
            dest_name = f"{true_g}_{vid_id}.mp4"
            shutil.copy2(mp4_path, demo_dir / dest_name)
            copied_rows.append((dest_name, true_g, pred_g, pred_conf))

    print("\n=== DEMO VIDEOS SUMMARY (VERIFIED VIA app.py TAB 2) ===")
    print(f"Total clips copied to demo_videos/: {len(copied_rows)}")
    print("| Filename | Ground-Truth Gloss | Predicted Gloss | Confidence (%) |")
    print("|---|---|---|---:|")
    for fname, true_g, pred_g, conf in copied_rows:
        print(f"| `{fname}` | `{true_g}` | `{pred_g}` | {conf*100:.2f}% |")

    print("\nVerified matches copied per gloss:")
    for g in TARGET_GLOSSES:
        print(f"  {g:<10}: {verified_counts[g]} clip(s)")


if __name__ == "__main__":
    main()
