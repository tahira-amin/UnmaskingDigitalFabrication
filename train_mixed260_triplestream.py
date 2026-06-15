import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import time
from tqdm import tqdm

# =============================================================================
# MATCHED-SIZE NO-BRANCHING BASELINE — TRAINER (STEP 3 of 3, TRIPLE-STREAM)
# -----------------------------------------------------------------------------
# Mirrors train_branch_a_triplestream_farneback_new.py exactly, but runs on
# the balanced 260-video mixed-quality dataset (65 per subfolder x 4).
#
# Streams:
#   1. Cached 1280-dim EfficientNet-B0 features.
#   2. Cached 960-dim MobileNetV3-Large features.
#   3. 5-dim Farneback optical flow stats (recomputed fresh every run).
# Final feature vector: 1280 + 960 + 5 = 2245-dim.
#
# This is the fair triple-stream no-branching comparison: same training size
# and same feature set as Branch A, so any accuracy difference is attributable
# to branching itself, not to data volume or feature choice.
# =============================================================================

EFFNET_FEATURES_PATH    = "mixed260_features_effnet_triple.npy"
MOBILENET_FEATURES_PATH = "mixed260_features_mobilenet_triple.npy"
METADATA_PATH           = "mixed260_metadata_labeled_triple.csv"

FLOW_OUTPUT = "mixed260_flow_farneback_triple.npy"
MODEL_OUT   = "model_mixed260_triplestream.pkl"

N_FOLDS      = 5
C_GRID       = [0.001, 0.01, 0.1, 1, 10, 100]
RANDOM_STATE = 44
FLOW_RESIZE  = 224


def build_prev_paths(metadata):
    """For each row, return the path of the previous frame in the same video
    (by frame_num). First frame uses second frame (forward difference)."""
    metadata = metadata.copy()
    metadata['frame_num'] = metadata['filename'].str.extract(r'(\d+)').astype(int)
    prev_paths = [None] * len(metadata)
    for video_name, group in metadata.groupby('video_name'):
        sorted_group = group.sort_values('frame_num')
        idxs  = sorted_group.index.tolist()
        paths = sorted_group['frame_path'].tolist()
        n = len(idxs)
        for i, orig in enumerate(idxs):
            if i == 0:
                prev_paths[orig] = paths[1] if n > 1 else paths[0]
            else:
                prev_paths[orig] = paths[i - 1]
    return prev_paths


def compute_optical_flow_stats(records, prev_paths, output_path):
    """Fresh Farneback flow stats from disk. Writes to output_path for
    inspection only — never read back as a cache."""
    stats = np.zeros((len(records), 5), dtype=np.float32)
    print(f"   Computing Farneback optical flow for {len(records)} frame pairs (fresh, no cache)...")
    for i, m in enumerate(tqdm(records, desc="optical flow")):
        try:
            curr_gray = cv2.imread(m['frame_path'], cv2.IMREAD_GRAYSCALE)
            if prev_paths[i] == m['frame_path']:
                continue
            prev_gray = cv2.imread(prev_paths[i], cv2.IMREAD_GRAYSCALE)
            if curr_gray is None or prev_gray is None:
                continue
            if curr_gray.shape != prev_gray.shape:
                prev_gray = cv2.resize(prev_gray, (curr_gray.shape[1], curr_gray.shape[0]))
            curr_r = cv2.resize(curr_gray, (FLOW_RESIZE, FLOW_RESIZE))
            prev_r = cv2.resize(prev_gray, (FLOW_RESIZE, FLOW_RESIZE))
            flow = cv2.calcOpticalFlowFarneback(
                prev_r, curr_r, None,
                pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                poly_n=5, poly_sigma=1.2, flags=0
            )
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            stats[i, 0] = float(np.mean(mag))
            stats[i, 1] = float(np.std(mag))
            stats[i, 2] = float(np.max(mag))
            stats[i, 3] = float(np.mean(np.cos(ang)))
            stats[i, 4] = float(np.mean(np.sin(ang)))
        except Exception as e:
            print(f"     skip flow for {m['frame_path']}: {e}")

    np.save(output_path, stats)
    print(f"   Wrote fresh optical flow to {output_path} (for inspection; not used as cache).")
    return stats


def video_acc(y_true, y_pred, groups):
    df = pd.DataFrame({'video': groups, 'true': y_true, 'pred': y_pred})
    g = df.groupby('video').agg(true=('true', 'first'),
                                pred=('pred', lambda x: x.mode()[0]))
    return accuracy_score(g['true'], g['pred'])


def select_C_inner(X_train_full, y_train, train_groups):
    """LEAKAGE-FREE C selection. Receives UNSCALED features, fits fresh
    StandardScaler per inner fold. Returns (best_c, best_inner_score)."""
    unique_train_groups = np.unique(train_groups)
    n_splits = min(3, len(unique_train_groups))

    if n_splits < 2:
        print("   Warning: not enough training videos for inner GroupKFold. Using C=1.0.")
        return 1.0, np.nan

    inner_cv = GroupKFold(n_splits=n_splits)
    c_scores = {}

    for c in C_GRID:
        fold_scores = []
        for inner_train_idx, inner_val_idx in inner_cv.split(X_train_full, y_train, train_groups):
            X_inner_train = X_train_full[inner_train_idx]
            X_inner_val   = X_train_full[inner_val_idx]
            y_inner_train = y_train[inner_train_idx]
            y_inner_val   = y_train[inner_val_idx]

            if len(np.unique(y_inner_train)) < 2 or len(np.unique(y_inner_val)) < 2:
                continue

            inner_scaler = StandardScaler()
            X_inner_train_scaled = inner_scaler.fit_transform(X_inner_train)
            X_inner_val_scaled   = inner_scaler.transform(X_inner_val)

            clf = LogisticRegression(C=c, random_state=RANDOM_STATE,
                                     class_weight='balanced', max_iter=2000)
            clf.fit(X_inner_train_scaled, y_inner_train)
            y_inner_pred = clf.predict(X_inner_val_scaled)
            fold_scores.append(accuracy_score(y_inner_val, y_inner_pred))

        c_scores[c] = np.mean(fold_scores) if fold_scores else -np.inf

    if all(score == -np.inf for score in c_scores.values()):
        print("   Warning: inner GroupKFold could not form valid two-class folds. Using C=1.0.")
        return 1.0, np.nan

    best_c = max(c_scores, key=c_scores.get)
    return best_c, c_scores[best_c]


def main():
    start = time.time()
    print("--- TRAIN MIXED-260 TRIPLE-STREAM (EffNet + MobileNetV3 + Farneback Flow, NO PCA, 5-FOLD CV) ---")
    print("--- Matched-size no-branching baseline: 65 videos per subfolder x 4 = 260 total ---")
    print("--- (leakage-free inner CV: fresh scaler per inner fold) ---\n")

    try:
        semantic_e = np.load(EFFNET_FEATURES_PATH)
        semantic_m = np.load(MOBILENET_FEATURES_PATH)
        metadata   = pd.read_csv(METADATA_PATH)
        print(f"Loaded EfficientNet:  {semantic_e.shape}")
        print(f"Loaded MobileNetV3:   {semantic_m.shape}")
        print(f"Metadata rows:        {len(metadata)}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run feature_extractor_mixed260_triplestream.py and "
              "branch_mixed260_triplestream.py first.")
        return

    if not (len(semantic_e) == len(semantic_m) == len(metadata)):
        print(f"Error: Row count mismatch — EffNet:{len(semantic_e)}, "
              f"MobileNet:{len(semantic_m)}, Metadata:{len(metadata)}")
        return

    if 'label' not in metadata.columns:
        print("Error: metadata needs a 'label' column.")
        return
    if 'frame_path' not in metadata.columns:
        print("Error: metadata needs a 'frame_path' column.")
        return

    metadata['target'] = metadata['label'].apply(
        lambda x: 1 if str(x).strip().lower() == 'ai' else 0
    )

    y_all   = metadata['target'].values
    groups  = metadata['video_name'].values
    records = metadata[['frame_path', 'filename', 'video_name']].to_dict('records')

    print("\nBuilding prev-frame map (forward diff for first frame of each video)...")
    prev_paths = build_prev_paths(metadata)

    print("Computing Farneback optical flow (FRESH, no cache load)...")
    flow_stats = compute_optical_flow_stats(records, prev_paths, FLOW_OUTPUT)
    print(f"   Flow shape: {flow_stats.shape}, "
          f"mean_mag avg={flow_stats[:,0].mean():.3f}, max_mag avg={flow_stats[:,2].mean():.3f}")

    full = np.hstack([semantic_e, semantic_m, flow_stats])
    print(f"\nCombined feature dim: {full.shape[1]} "
          f"(effnet {semantic_e.shape[1]} + mobilenet {semantic_m.shape[1]} + "
          f"flow {flow_stats.shape[1]})")

    print(f"\nRunning {N_FOLDS}-fold video-level cross-validation (NO PCA)...")
    gkf = GroupKFold(n_splits=N_FOLDS)
    frame_accs, vid_accs, fold_Cs, fold_inner_scores = [], [], [], []
    fold_num = 0

    for train_idx, test_idx in gkf.split(full, y_all, groups):
        fold_num += 1
        X_tr_full, X_te_full = full[train_idx], full[test_idx]
        y_tr, y_te = y_all[train_idx], y_all[test_idx]
        groups_tr, groups_te = groups[train_idx], groups[test_idx]

        print(f"\n=== FOLD {fold_num}/{N_FOLDS} ===")
        print(f"Train: {len(train_idx)} frames / {len(np.unique(groups_tr))} videos | "
              f"Test: {len(test_idx)} frames / {len(np.unique(groups_te))} videos")

        best_c, best_inner_score = select_C_inner(X_tr_full, y_tr, groups_tr)
        fold_Cs.append(best_c)
        fold_inner_scores.append(best_inner_score)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_full)
        X_te = scaler.transform(X_te_full)

        clf = LogisticRegression(C=best_c, random_state=RANDOM_STATE,
                                 class_weight='balanced', max_iter=2000)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)

        f_acc = accuracy_score(y_te, y_pred)
        v_acc = video_acc(y_te, y_pred, groups_te)
        frame_accs.append(f_acc); vid_accs.append(v_acc)
        inner_str = f"{best_inner_score:.4f}" if not np.isnan(best_inner_score) else "n/a"
        print(f"  Fold {fold_num}: C={best_c}  inner_cv={inner_str}  "
              f"frame={f_acc:.4f}  video={v_acc:.4f}")

    print("\n" + "="*70)
    print("5-FOLD CV RESULTS (Mixed-260 Triple-Stream: EffNet + MobileNetV3 + Flow)")
    print("="*70)
    print(f"Per-fold C values:        {fold_Cs}")
    print(f"Per-fold inner CV scores: {[f'{s:.4f}' if not np.isnan(s) else 'n/a' for s in fold_inner_scores]}")
    print(f"Per-fold FRAME accuracy:  {[f'{a:.4f}' for a in frame_accs]}")
    print(f"Per-fold VIDEO accuracy:  {[f'{a:.4f}' for a in vid_accs]}")
    print(f"\nAVERAGE Frame Accuracy: {np.mean(frame_accs):.4f} +/- {np.std(frame_accs):.4f}")
    print(f"AVERAGE Video Accuracy: {np.mean(vid_accs):.4f} +/- {np.std(vid_accs):.4f}")

    # ---- FINAL DEPLOYABLE MODEL ----
    print("\nFitting FINAL deployable model on ALL data...")
    best_c_final, final_inner_score = select_C_inner(full, y_all, groups)
    inner_str = f"{final_inner_score:.4f}" if not np.isnan(final_inner_score) else "n/a"
    print(f"   Final selected C: {best_c_final}  (inner cv = {inner_str})")

    scaler_final = StandardScaler()
    X_all = scaler_final.fit_transform(full)
    final_clf = LogisticRegression(C=best_c_final, random_state=RANDOM_STATE,
                                   class_weight='balanced', max_iter=2000)
    final_clf.fit(X_all, y_all)

    joblib.dump({
        'model': final_clf,
        'scaler': scaler_final,
        'flow_output_for_inspection': FLOW_OUTPUT,
        'feature_layout': '1280 EfficientNet + 960 MobileNetV3 + 5 Farneback flow = 2245 dim',
        'cv_frame_accuracy_mean': float(np.mean(frame_accs)),
        'cv_frame_accuracy_std':  float(np.std(frame_accs)),
        'cv_video_accuracy_mean': float(np.mean(vid_accs)),
        'cv_video_accuracy_std':  float(np.std(vid_accs)),
        'cv_per_fold_C': fold_Cs,
        'cv_per_fold_inner_scores': fold_inner_scores,
    }, MODEL_OUT)
    print(f"Saved model to {MODEL_OUT}")
    print(f"\nTotal time: {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
