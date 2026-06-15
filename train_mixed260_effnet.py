import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import time

# =============================================================================
# MATCHED-SIZE NO-BRANCHING BASELINE — TRAINER (STEP 3 of 3)
# -----------------------------------------------------------------------------
# Same pipeline as train_branch_a_v4_nopca.py. Run on the balanced 260-video
# mixed-quality dataset. This is the fair no-branching comparison — same
# training size as each individual branch, so any accuracy difference is
# attributable to branching itself, not to data volume.
#
# CHANGES vs the previous version of this file:
#   * Temporal frame-diff features removed (now uses CNN features only).
#   * select_C_inner is now LEAKAGE-FREE. It receives UNSCALED training
#     features and fits a fresh StandardScaler per inner fold. Previously it
#     received already-scaled features, so the outer scaler statistics
#     (computed on the FULL outer train) reached the inner-validation rows.
# =============================================================================

# CONFIGURATION
FEATURES_PATH = "mixed260_features_effnet.npy"
METADATA_PATH = "mixed260_metadata_labeled_effnet.csv"
MODEL_OUT = "model_mixed260_effnet.pkl"

# NO PCA. Removed due to instability; L2-regularized LogisticRegression
# (C tuned by inner CV) handles full dimensionality directly.

N_FOLDS = 5
C_GRID  = [0.001, 0.01, 0.1, 1, 10, 100]
RANDOM_STATE = 42

def video_acc(y_true, y_pred, groups):
    df = pd.DataFrame({'video': groups, 'true': y_true, 'pred': y_pred})
    g = df.groupby('video').agg(true=('true', 'first'),
                                pred=('pred', lambda x: x.mode()[0]))
    return accuracy_score(g['true'], g['pred'])

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

def main():
    start_time = time.time()

    print("--- TRAIN MIXED-260 (EfficientNet Features + Logistic Regression, NO PCA, 5-FOLD CV) ---")
    print("--- (no temporal features; leakage-free inner CV) ---")
    print("    Matched-size no-branching baseline: 65 videos per subfolder x 4 = 260 total.\n")

    try:
        features = np.load(FEATURES_PATH)
        metadata = pd.read_csv(METADATA_PATH)
        print(f"Loaded {len(features)} frames with {features.shape[1]}-dim features")
    except FileNotFoundError:
        print(f"Error: Files not found. Run feature_extractor_mixed260_effnet.py and branch_mixed260_effnet.py first.")
        return

    if len(features) != len(metadata):
        print("Error: Feature/metadata mismatch")
        return

    y_all  = metadata['target'].values
    groups = metadata['video_name'].values

    print(f"\nRunning {N_FOLDS}-fold video-level cross-validation (NO PCA)...")
    gkf = GroupKFold(n_splits=N_FOLDS)

    frame_accs, vid_accs, fold_Cs, fold_inner_scores = [], [], [], []
    fold_num   = 0

    for train_idx, test_idx in gkf.split(features, y_all, groups):
        fold_num += 1
        X_tr_full, X_te_full = features[train_idx], features[test_idx]
        y_tr, y_te = y_all[train_idx], y_all[test_idx]
        groups_tr  = groups[train_idx]
        groups_te  = groups[test_idx]

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
        clf.fit(X_tr, y_tr)

        y_pred = clf.predict(X_te)

        f_acc = accuracy_score(y_te, y_pred)
        v_acc = video_acc(y_te, y_pred, groups_te)
        frame_accs.append(f_acc)
        vid_accs.append(v_acc)

        inner_str = f"{best_inner_score:.4f}" if not np.isnan(best_inner_score) else "n/a"
        print(f"   Fold {fold_num}: C={best_c}  inner_cv={inner_str}  frame={f_acc:.4f}  video={v_acc:.4f}")

    print("\n" + "="*70)
    print("5-FOLD CROSS-VALIDATION RESULTS (NO PCA, MIXED-260)")
    print("="*70)
    print(f"Per-fold C values:        {fold_Cs}")
    print(f"Per-fold inner CV scores: {[f'{s:.4f}' if not np.isnan(s) else 'n/a' for s in fold_inner_scores]}")
    print(f"Per-fold FRAME accuracy:  {[f'{a:.4f}' for a in frame_accs]}")
    print(f"Per-fold VIDEO accuracy:  {[f'{a:.4f}' for a in vid_accs]}")
    print(f"\nAVERAGE Frame Accuracy: {np.mean(frame_accs):.4f} +/- {np.std(frame_accs):.4f}")
    print(f"AVERAGE Video Accuracy: {np.mean(vid_accs):.4f} +/- {np.std(vid_accs):.4f}")

    print("\nFitting final model on ALL data for saving...")

    # C selection on UNSCALED features for the final model too.
    best_c_final, final_inner_score = select_C_inner(features, y_all, groups)
    inner_str = f"{final_inner_score:.4f}" if not np.isnan(final_inner_score) else "n/a"
    print(f"   Final selected C: {best_c_final}  (inner cv = {inner_str})")

    scaler_final = StandardScaler()
    X_all = scaler_final.fit_transform(features)
    final_model = LogisticRegression(C=best_c_final, random_state=RANDOM_STATE,
                                     class_weight='balanced', max_iter=2000)
    final_model.fit(X_all, y_all)

    joblib.dump({
        'model': final_model,
        'scaler': scaler_final
    }, MODEL_OUT)
    print(f"\nModel saved to {MODEL_OUT}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total Execution Time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()