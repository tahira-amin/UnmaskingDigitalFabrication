import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, average_precision_score, roc_auc_score)
import time
from tqdm import tqdm


# SINGLE SPLIT TRAINER (Branch A) — EffNet + MobileNetV3 + FARNEBACK FLOW
# -----------------------------------------------------------------------------
# Identical to the 5-fold CV trainer in every way EXCEPT that it runs a
# single GroupShuffleSplit (80% train / 20% test, grouped by video_name)
# instead of 5-fold GroupKFold. Use this for fast sklearn version testing.
# Everything else — C selection, leakage-free inner CV, StandardScaler,
# LogReg, mode voting, metrics — is unchanged.

EFFNET_FEATURES_PATH    = "branch_a_features_efficientnet.npy"
EFFNET_METADATA_PATH    = "branch_a_metadata_labeled_efficientnet.csv"

MOBILENET_FEATURES_PATH = "branch_a_features_mobilenet.npy"
MOBILENET_METADATA_PATH = "branch_a_metadata_labeled_mobilenet.csv"

FLOW_OUTPUT             = "branch_a_flow_farneback_triple.npy"
MODEL_OUT               = "model_branch_a_single_split.pkl"

TEST_SIZE    = 0.20        # 20% of videos go to test (mirrors one 5-fold split)
C_GRID       = [0.001, 0.01, 0.1, 1, 10, 100]
RANDOM_STATE = 42
FLOW_RESIZE  = 224


def build_prev_paths(metadata):
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


def video_level_preds(y_true, y_pred, groups):
    df = pd.DataFrame({'video': groups, 'true': y_true, 'pred': y_pred})
    g = df.groupby('video').agg(true=('true', 'first'),
                                pred=('pred', lambda x: x.mode()[0]))
    return g['true'].values, g['pred'].values


def select_C_inner(X_train_full, y_train, train_groups):
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
    print("--- TRAIN BRANCH A SINGLE SPLIT (EffNet + MobileNetV3 + Farneback Flow) ---")
    print(f"--- 80/20 GroupShuffleSplit by video_name | random_state={RANDOM_STATE} ---\n")

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

    if len(semantic_e) != len(meta_e):
        print(f"Error: EffNet feature/metadata mismatch ({len(semantic_e)} vs {len(meta_e)})")
        return
    if len(semantic_m) != len(meta_m):
        print(f"Error: MobileNet feature/metadata mismatch ({len(semantic_m)} vs {len(meta_m)})")
        return

    aligned, msg = verify_alignment(meta_e, meta_m)
    if not aligned:
        print(f"\nError: EffNet and MobileNet metadata DO NOT align: {msg}")
        return
    print("Verified: EffNet and MobileNet metadata are row-aligned.\n")

    metadata = meta_e.copy()

    if 'label' not in metadata.columns:
        print("Error: metadata needs a 'label' column.")
        return
    if 'frame_path' not in metadata.columns:
        print("Error: metadata needs 'frame_path'.")
        return

    metadata['target'] = metadata['label'].apply(lambda x: 1 if str(x).strip().lower() == 'ai' else 0)

    y_all   = metadata['target'].values
    groups  = metadata['video_name'].values
    records = metadata[['frame_path', 'filename', 'video_name']].to_dict('records')

    print("Building prev-frame map (forward diff for first frame of each video)...")
    prev_paths = build_prev_paths(metadata)

    print("Computing Farneback optical flow (FRESH, no cache load)...")
    t0_flow = time.time()
    flow_stats = compute_optical_flow_stats(records, prev_paths, FLOW_OUTPUT)
    flow_time = time.time() - t0_flow
    print(f"   Flow shape: {flow_stats.shape}, "
          f"mean_mag avg={flow_stats[:,0].mean():.3f}, max_mag avg={flow_stats[:,2].mean():.3f}")

    full = np.hstack([semantic_e, semantic_m, flow_stats])
    print(f"\nCombined feature dim: {full.shape[1]} "
          f"(effnet {semantic_e.shape[1]} + mobilenet {semantic_m.shape[1]} + flow {flow_stats.shape[1]})")

    # --- Single GroupShuffleSplit ---
    print(f"\nRunning single GroupShuffleSplit (test_size={TEST_SIZE}, random_state={RANDOM_STATE})...")

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(full, y_all, groups))

    X_tr_full, X_te_full = full[train_idx], full[test_idx]
    y_tr, y_te = y_all[train_idx], y_all[test_idx]
    groups_tr, groups_te = groups[train_idx], groups[test_idx]

    print(f"Train: {len(train_idx)} frames / {len(np.unique(groups_tr))} videos")
    print(f"Test:  {len(test_idx)} frames / {len(np.unique(groups_te))} videos")

    # C selection on UNSCALED features
    best_c, best_inner_score = select_C_inner(X_tr_full, y_tr, groups_tr)
    inner_str = f"{best_inner_score:.4f}" if not np.isnan(best_inner_score) else "n/a"
    print(f"\nSelected C={best_c}  (inner CV score = {inner_str})")

    # Outer scaler — fit on train only
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
    test_time = time.time() - t0_test

    print("\n" + "="*70)
    print("SINGLE SPLIT RESULTS (EfficientNet + MobileNetV3 + Optical Flow)")
    print("="*70)
    print(f"Selected C:              {best_c}  (inner CV = {inner_str})")
    print(f"\nFrame Accuracy:          {f_acc:.4f}")
    print(f"Frame Precision:         {f_prec:.4f}")
    print(f"Frame Recall:            {f_rec:.4f}")
    print(f"Frame F1:                {f_f1:.4f}")
    print(f"Frame AP:                {f_ap:.4f}")
    print(f"Frame ROC-AUC:           {f_auc:.4f}")
    print(f"\nVideo Accuracy:          {v_acc:.4f}")
    print(f"Video Precision:         {v_prec:.4f}")
    print(f"Video Recall:            {v_rec:.4f}")
    print(f"Video F1:                {v_f1:.4f}")
    total_train_time = flow_time + train_time
    print(f"\nOptical Flow Time:       {flow_time:.1f}s")
    print(f"SVM Training Time:       {train_time:.3f}s")
    print(f"Total Training Time:     {total_train_time:.1f}s")
    print(f"Inference time:          {infer_time:.4f}s")
    print(f"Test eval time:          {test_time:.4f}s")

    print(f"\nTotal script time: {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()