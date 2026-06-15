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
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, average_precision_score, roc_auc_score)
import joblib
import time
from tqdm import tqdm

# =============================================================================
# TRIPLE-STREAM TRAINER (Branch A) — EffNet + MobileNetV3 + FARNEBACK FLOW
# -----------------------------------------------------------------------------
# Streams:
#   1. Cached 1280-dim EfficientNet-B0 semantic features.
#   2. Cached 960-dim MobileNetV3-Large semantic features.
#   3. 5-dim Farneback optical flow stats per frame (mean_mag, std_mag,
#      max_mag, mean_cos_angle, mean_sin_angle).
# Final feature vector: 1280 + 960 + 5 = 2245-dim.
# No noise CNN. No temporal frame-diff stats. No PCA.
# 5-fold GroupKFold, StandardScaler-on-train-only, inner-CV C selection.
#
# CHANGE vs the previous version of this file:
#   * select_C_inner is now LEAKAGE-FREE. It receives UNSCALED training
#     features and fits a fresh StandardScaler per inner fold. Previously it
#     received already-scaled features, which let the outer scaler statistics
#     (computed on the FULL outer train) reach the inner-validation rows.
#     Leak was scaler-only (no labels, no values) and symmetric across C,
#     but it's still cleaner to remove it.
#
# CACHE POLICY: optical flow is recomputed FRESHLY on every run. Written to
# FLOW_OUTPUT for inspection but never read back.
#
# Both .npy files MUST come from the same frame set in the same order. The
# script verifies this row-by-row using the (video_name, filename) pair in
# the two metadata CSVs.
# =============================================================================

EFFNET_FEATURES_PATH    = "branch_a_features_dualstream_fa.npy"               # 1280-dim
EFFNET_METADATA_PATH    = "branch_a_metadata_labeled_dualstream_fa.csv"

MOBILENET_FEATURES_PATH = "branch_a_features_dualstream_fa_mobilenet.npy"     # 960-dim
MOBILENET_METADATA_PATH = "branch_a_metadata_labeled_dualstream_fa_mobilenet.csv"

FLOW_OUTPUT             = "branch_a_flow_farneback_triple.npy"                # written, not read
MODEL_OUT               = "model_branch_a_triplestream_farneback.pkl"

N_FOLDS      = 5
C_GRID       = [0.001, 0.01, 0.1, 1, 10, 100]
RANDOM_STATE = 42
FLOW_RESIZE  = 224


def build_prev_paths(metadata):
    """For each row in metadata, return the path of the previous frame in the
    same video (by frame_num). FIRST frame of a video uses the SECOND frame as
    reference (forward difference). Single-frame videos use themselves."""
    metadata = metadata.copy()
    metadata['frame_num'] = metadata['filename'].str.extract(r'(\d+)').astype(int)
    prev_paths = [None] * len(metadata)
    for video_name, group in metadata.groupby('video_name'):
        sorted_group = group.sort_values('frame_num')
        idxs   = sorted_group.index.tolist()
        paths  = sorted_group['frame_path'].tolist()
        n = len(idxs)
        for i, orig in enumerate(idxs):
            if i == 0:
                prev_paths[orig] = paths[1] if n > 1 else paths[0]
            else:
                prev_paths[orig] = paths[i - 1]
    return prev_paths


def compute_optical_flow_stats(records, prev_paths, output_path):
    """ALWAYS recompute Farneback flow stats from disk. Writes the result to
    output_path for inspection but does not read it back."""
    stats = np.zeros((len(records), 5), dtype=np.float32)
    print(f"   Computing Farneback optical flow for {len(records)} frame pairs (fresh, no cache)...")
    for i, m in enumerate(tqdm(records, desc="optical flow")):
        try:
            curr_gray = cv2.imread(m['frame_path'], cv2.IMREAD_GRAYSCALE)
            if prev_paths[i] == m['frame_path']:
                continue  # single-frame video edge case -> stays zero
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


def video_level_preds(y_true, y_pred, groups):
    df = pd.DataFrame({'video': groups, 'true': y_true, 'pred': y_pred})
    g = df.groupby('video').agg(true=('true', 'first'),
                                pred=('pred', lambda x: x.mode()[0]))
    return g['true'].values, g['pred'].values


def video_auc(y_true, y_proba, groups):
    """AUC computed at video level using mean per-frame probability per video.
    This is more honest than frame AUC because it treats each VIDEO as one
    independent sample rather than inflating sample size with correlated frames."""
    df = pd.DataFrame({'video': groups, 'true': y_true, 'proba': y_proba})
    g = df.groupby('video').agg(true=('true', 'first'),
                                proba=('proba', 'mean'))
    if len(np.unique(g['true'].values)) < 2:
        return np.nan
    return roc_auc_score(g['true'].values, g['proba'].values)


def select_C_inner(X_train_full, y_train, train_groups):
    """LEAKAGE-FREE C selection. Receives UNSCALED training features and fits
    a fresh StandardScaler per inner fold. Scaler statistics never cross the
    inner train/val boundary.

    Returns (best_c, best_inner_score) so the caller can print the inner CV
    score next to the chosen C for diagnostics.
    """
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
    best_score = c_scores[best_c]
    return best_c, best_score


def verify_alignment(meta_e, meta_m):
    """Check that the two metadata files describe the same frames in the same
    row order. Compares (video_name, filename) tuples row-by-row."""
    if len(meta_e) != len(meta_m):
        return False, f"row count mismatch ({len(meta_e)} vs {len(meta_m)})"
    for col in ('video_name', 'filename'):
        if col not in meta_e.columns or col not in meta_m.columns:
            return False, f"both metadata files must have a '{col}' column"
    diff_mask = (meta_e['video_name'].values != meta_m['video_name'].values) | \
                (meta_e['filename'].values   != meta_m['filename'].values)
    n_diff = int(diff_mask.sum())
    if n_diff > 0:
        first = np.argmax(diff_mask)
        return False, (f"{n_diff} rows disagree between metadata files; "
                       f"first mismatch at row {first}: "
                       f"effnet=({meta_e.iloc[first]['video_name']}, {meta_e.iloc[first]['filename']}) "
                       f"mobilenet=({meta_m.iloc[first]['video_name']}, {meta_m.iloc[first]['filename']})")
    return True, "OK"


def main():
    start = time.time()
    print("--- TRAIN BRANCH A TRIPLE-STREAM (EffNet + MobileNetV3 + Farneback Flow, FRESH STATS, NO PCA, 5-FOLD CV) ---")
    print("--- (leakage-free inner CV: fresh scaler per inner fold) ---\n")

    # --- Load EfficientNet stream ---
    try:
        semantic_e = np.load(EFFNET_FEATURES_PATH)
        meta_e     = pd.read_csv(EFFNET_METADATA_PATH)
        print(f"Loaded EfficientNet: {semantic_e.shape}  metadata rows: {len(meta_e)}")
    except FileNotFoundError:
        print(f"Error: {EFFNET_FEATURES_PATH} or {EFFNET_METADATA_PATH} not found.")
        return

    # --- Load MobileNetV3 stream ---
    try:
        semantic_m = np.load(MOBILENET_FEATURES_PATH)
        meta_m     = pd.read_csv(MOBILENET_METADATA_PATH)
        print(f"Loaded MobileNetV3: {semantic_m.shape}  metadata rows: {len(meta_m)}")
    except FileNotFoundError:
        print(f"Error: {MOBILENET_FEATURES_PATH} or {MOBILENET_METADATA_PATH} not found.")
        return

    # --- Verify row counts match across files ---
    if len(semantic_e) != len(meta_e):
        print(f"Error: EffNet feature/metadata mismatch ({len(semantic_e)} vs {len(meta_e)})")
        return
    if len(semantic_m) != len(meta_m):
        print(f"Error: MobileNet feature/metadata mismatch ({len(semantic_m)} vs {len(meta_m)})")
        return

    # --- Verify the two metadata files describe the same frames in the same order ---
    aligned, msg = verify_alignment(meta_e, meta_m)
    if not aligned:
        print(f"\nError: EffNet and MobileNet metadata DO NOT align: {msg}")
        print("Both extractors must process the same frames in the same order.")
        return
    print("Verified: EffNet and MobileNet metadata are row-aligned.\n")

    # Use EffNet metadata as the canonical reference (labels + frame paths).
    metadata = meta_e.copy()

    if 'label' not in metadata.columns:
        print("Error: metadata needs a 'label' column.")
        return
    if 'frame_path' not in metadata.columns:
        print("Error: metadata needs 'frame_path'.")
        return

    # LABEL-SAFE: always derive target from label
    metadata['target'] = metadata['label'].apply(lambda x: 1 if str(x).strip().lower() == 'ai' else 0)

    y_all   = metadata['target'].values
    groups  = metadata['video_name'].values
    records = metadata[['frame_path', 'filename', 'video_name']].to_dict('records')

    # --- Build prev-frame map (fresh on every run) ---
    print("Building prev-frame map (forward diff for first frame of each video)...")
    prev_paths = build_prev_paths(metadata)

    # --- Compute optical flow stats FRESHLY (no cache load) ---
    print("Computing Farneback optical flow (FRESH, no cache load)...")
    t0_flow = time.time()
    flow_stats = compute_optical_flow_stats(records, prev_paths, FLOW_OUTPUT)
    flow_time = time.time() - t0_flow
    print(f"   Flow shape: {flow_stats.shape}, "
          f"mean_mag avg={flow_stats[:,0].mean():.3f}, max_mag avg={flow_stats[:,2].mean():.3f}")

    # --- Combine streams: effnet (1280) + mobilenet (960) + flow (5) = 2245 dims ---
    full = np.hstack([semantic_e, semantic_m, flow_stats])
    print(f"\nCombined feature dim: {full.shape[1]} "
          f"(effnet {semantic_e.shape[1]} + mobilenet {semantic_m.shape[1]} + flow {flow_stats.shape[1]})")

    print(f"\nRunning {N_FOLDS}-fold video-level cross-validation (NO PCA)...")
    gkf = GroupKFold(n_splits=N_FOLDS)
    frame_accs, vid_accs, fold_Cs, fold_inner_scores = [], [], [], []
    frame_precs, frame_recs, frame_f1s = [], [], []
    vid_precs, vid_recs, vid_f1s = [], [], []
    frame_aps, frame_aucs, vid_aucs = [], [], []
    infer_times, n_test_vids_list = [], []
    fold_num = 0
    cv_start = time.time()

    for train_idx, test_idx in gkf.split(full, y_all, groups):
        fold_num += 1
        X_tr_full, X_te_full = full[train_idx], full[test_idx]
        y_tr, y_te = y_all[train_idx], y_all[test_idx]
        groups_tr, groups_te = groups[train_idx], groups[test_idx]

        print(f"\n=== FOLD {fold_num}/{N_FOLDS} ===")
        print(f"Train: {len(train_idx)} frames / {len(np.unique(groups_tr))} videos | "
              f"Test: {len(test_idx)} frames / {len(np.unique(groups_te))} videos")

        # C selection on UNSCALED features (fresh scaler per inner fold).
        best_c, best_inner_score = select_C_inner(X_tr_full, y_tr, groups_tr)
        fold_Cs.append(best_c)
        fold_inner_scores.append(best_inner_score)

        # Outer scaler — fit on train only, transform test.
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_full)
        X_te = scaler.transform(X_te_full)

        clf = LogisticRegression(C=best_c, random_state=RANDOM_STATE,
                                 class_weight='balanced', max_iter=2000)
        t0_train = time.time()
        clf.fit(X_tr, y_tr)
        train_time = time.time() - t0_train

        t0_infer = time.time()
        y_pred = clf.predict(X_te)
        infer_time = time.time() - t0_infer

        t0_test = time.time()
        y_proba = clf.predict_proba(X_te)[:, 1]
        f_acc  = accuracy_score(y_te, y_pred)
        f_prec = precision_score(y_te, y_pred, zero_division=0)
        f_rec  = recall_score(y_te, y_pred, zero_division=0)
        f_f1   = f1_score(y_te, y_pred, zero_division=0)
        f_ap   = average_precision_score(y_te, y_proba)
        f_auc  = roc_auc_score(y_te, y_proba)
        vt, vp = video_level_preds(y_te, y_pred, groups_te)
        v_acc  = accuracy_score(vt, vp)
        v_prec = precision_score(vt, vp, zero_division=0)
        v_rec  = recall_score(vt, vp, zero_division=0)
        v_f1   = f1_score(vt, vp, zero_division=0)
        v_auc  = video_auc(y_te, y_proba, groups_te)
        test_time = time.time() - t0_test

        frame_accs.append(f_acc);  vid_accs.append(v_acc)
        frame_precs.append(f_prec); frame_recs.append(f_rec); frame_f1s.append(f_f1)
        vid_precs.append(v_prec);  vid_recs.append(v_rec);  vid_f1s.append(v_f1)
        frame_aps.append(f_ap);    frame_aucs.append(f_auc); vid_aucs.append(v_auc)
        n_test_vids = len(np.unique(groups_te))
        infer_times.append(infer_time)
        n_test_vids_list.append(n_test_vids)

        inner_str = f"{best_inner_score:.4f}" if not np.isnan(best_inner_score) else "n/a"
        print(f"  Fold {fold_num}: C={best_c}  inner_cv={inner_str}  "
              f"frame={f_acc:.4f}  video={v_acc:.4f}")
        print(f"           frame prec={f_prec:.4f} rec={f_rec:.4f} f1={f_f1:.4f} "
              f"AP={f_ap:.4f} AUC={f_auc:.4f}")
        v_auc_str = f"{v_auc:.4f}" if not np.isnan(v_auc) else "n/a"
        print(f"           video prec={v_prec:.4f} rec={v_rec:.4f} f1={v_f1:.4f} AUC={v_auc_str}")
        print(f"           infer_per_video={infer_time/n_test_vids:.4f}s")

    print("\n" + "="*70)
    print("5-FOLD CV RESULTS (EfficientNet + MobileNetV3 + Optical Flow, FRESH STATS)")
    print("="*70)
    print(f"Per-fold C values:        {fold_Cs}")
    print(f"Per-fold inner CV scores: {[f'{s:.4f}' if not np.isnan(s) else 'n/a' for s in fold_inner_scores]}")
    print(f"Per-fold FRAME accuracy:  {[f'{a:.4f}' for a in frame_accs]}")
    print(f"Per-fold VIDEO accuracy:  {[f'{a:.4f}' for a in vid_accs]}")
    print(f"\nAVERAGE Frame Accuracy:   {np.mean(frame_accs):.4f} +/- {np.std(frame_accs):.4f}")
    print(f"AVERAGE Frame Precision:  {np.mean(frame_precs):.4f} +/- {np.std(frame_precs):.4f}")
    print(f"AVERAGE Frame Recall:     {np.mean(frame_recs):.4f} +/- {np.std(frame_recs):.4f}")
    print(f"AVERAGE Frame F1:         {np.mean(frame_f1s):.4f} +/- {np.std(frame_f1s):.4f}")
    print(f"AVERAGE Frame AP:         {np.mean(frame_aps):.4f} +/- {np.std(frame_aps):.4f}")
    print(f"Mean AP (avg over folds): {np.mean(frame_aps):.4f}")
    print(f"AVERAGE Frame ROC-AUC:    {np.mean(frame_aucs):.4f} +/- {np.std(frame_aucs):.4f}")
    print(f"\nAVERAGE Video Accuracy:   {np.mean(vid_accs):.4f} +/- {np.std(vid_accs):.4f}")
    print(f"AVERAGE Video Precision:  {np.mean(vid_precs):.4f} +/- {np.std(vid_precs):.4f}")
    print(f"AVERAGE Video Recall:     {np.mean(vid_recs):.4f} +/- {np.std(vid_recs):.4f}")
    print(f"AVERAGE Video F1:         {np.mean(vid_f1s):.4f} +/- {np.std(vid_f1s):.4f}")
    valid_vid_aucs = [a for a in vid_aucs if not np.isnan(a)]
    if valid_vid_aucs:
        print(f"AVERAGE Video ROC-AUC:    {np.mean(valid_vid_aucs):.4f} +/- {np.std(valid_vid_aucs):.4f}")
    cv_total_time = time.time() - cv_start
    print(f"\n5-Fold CV Training Time:       {cv_total_time:.1f}s")

    # ---- FINAL DEPLOYABLE MODEL ----
    print("\nFitting FINAL deployable model on ALL data...")
    # C selection on UNSCALED features for the final model too.
    best_c_final, final_inner_score = select_C_inner(full, y_all, groups)
    inner_str = f"{final_inner_score:.4f}" if not np.isnan(final_inner_score) else "n/a"
    print(f"   Final selected C: {best_c_final}  (inner cv = {inner_str})")

    scaler_final = StandardScaler()
    X_all = scaler_final.fit_transform(full)
    final_clf = LogisticRegression(C=best_c_final, random_state=RANDOM_STATE,
                                   class_weight='balanced', max_iter=2000)
    t0_final = time.time()
    final_clf.fit(X_all, y_all)
    final_train_time = time.time() - t0_final

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

    # --- Model Statistics ---
    n_features  = full.shape[1]
    lr_params   = final_clf.coef_.size + final_clf.intercept_.size
    lr_size_kb  = (lr_params * 8) / 1024          # float64 coefficients
    lr_flops    = 2 * n_features + 1               # dot-product + bias per sample

    # single-sample LogReg inference (warm-up then timed loop)
    _x1 = X_all[[0]]
    for _ in range(20):
        final_clf.predict(_x1)
    _t0 = time.perf_counter()
    for _ in range(1000):
        final_clf.predict(_x1)
    lr_infer_ms = (time.perf_counter() - _t0) / 1000 * 1000

    print("\n" + "="*70)
    print("MODEL STATISTICS")
    print("="*70)
    infer_per_video = sum(infer_times) / sum(n_test_vids_list)
    print(f"Optical Flow Time:               {flow_time:.1f}s")
    print(f"5-Fold CV Training Time:         {cv_total_time:.1f}s")
    print(f"Final Model Training Time:       {final_train_time:.3f}s")
    print(f"Inference Time per Video:        {infer_per_video*1000:.2f}ms")
    print(f"Feature dimensions:              {n_features:,}")
    print(f"LogReg — parameters:             {lr_params:,}")
    print(f"LogReg — model size:             {lr_size_kb:.2f} KB")
    print(f"LogReg — FLOPs/sample:           {lr_flops:,}")
    print(f"LogReg — single-sample infer:    {lr_infer_ms:.4f} ms")

    print(f"\nTotal script time: {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()