import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import torch


#  FORCE EXACT PRECISION FOR SRM NOISE EXTRACTION ON CUDA

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
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
import random
from tqdm import tqdm


# TRIPLE-STREAM TRAINER (Branch B) — MobileNetV3 + Noise CNN + Farneback Flow

# Streams per frame:
#   1. Cached 960-dim MobileNetV3-Large semantic features.
#   2. 128-dim noise features from a FOLD-AWARE noise CNN (SRM + 3 conv).
#      A fresh noise CNN is trained on each fold's TRAIN videos only.
#   3. 5-dim Farneback optical flow stats (mean_mag, std_mag, max_mag,
#      mean_cos_angle, mean_sin_angle) per frame.
# Final feature vector: 960 + 128 + 5 = 1093-dim.

# Inner CV: LEAKAGE-FREE. select_C_inner receives UNSCALED features and fits
# a fresh StandardScaler per inner fold.
# =============================================================================

SEMANTIC_FEATURES_PATH = "branch_b_features_mobilenet.npy"
METADATA_PATH          = "branch_b_metadata_labeled_mobilenet.csv"
FLOW_OUTPUT            = "branch_b_flow_farneback_triplestream.npy"
MODEL_OUT              = "model_branch_b_triplestream_noise_flow.pkl"

# Noise CNN fine-tune config
BATCH_TRAIN     = 64
BATCH_EXTRACT   = 32
FINETUNE_EPOCHS = 5
FINETUNE_LR     = 5e-4
WEIGHT_DECAY    = 5e-4

# CV / logreg config
N_FOLDS      = 5
C_GRID       = [0.001, 0.01, 0.1, 1, 10, 100]
RANDOM_STATE = 42
FLOW_RESIZE  = 224

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# --- SRM kernels (Fridrich & Kodovsky 2012) — fixed, non-trainable ---
SRM_KERNELS = np.array([
    [[ 0, 0, 0, 0, 0],[ 0, 0, 0, 0, 0],[ 0, 1,-2, 1, 0],[ 0, 0, 0, 0, 0],[ 0, 0, 0, 0, 0]],
    [[ 0, 0, 0, 0, 0],[ 0,-1, 2,-1, 0],[ 0, 2,-4, 2, 0],[ 0,-1, 2,-1, 0],[ 0, 0, 0, 0, 0]],
    [[-1, 2,-2, 2,-1],[ 2,-6, 8,-6, 2],[-2, 8,-12,8,-2],[ 2,-6, 8,-6, 2],[-1, 2,-2, 2,-1]],
], dtype=np.float32)
SRM_NORMS = np.array([2.0, 4.0, 12.0], dtype=np.float32)


class NoiseStreamCNN(nn.Module):
    """SRM high-pass filter (fixed) + shallow 3-conv CNN -> 128-dim noise features."""
    def __init__(self, out_dim=128):
        super().__init__()
        weight = np.zeros((9, 3, 5, 5), dtype=np.float32)
        for k in range(3):
            for c in range(3):
                weight[k*3+c, c] = SRM_KERNELS[k] / SRM_NORMS[k]
        self.srm = nn.Conv2d(3, 9, kernel_size=5, padding=2, bias=False)
        self.srm.weight.data = torch.from_numpy(weight)
        for p in self.srm.parameters():
            p.requires_grad = False

        self.conv1 = nn.Conv2d(9, 32, 3, stride=2, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, out_dim, 3, stride=2, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_dim)
        self.gap   = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.srm(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.gap(x).flatten(1)


class FrameDataset(Dataset):
    def __init__(self, records, preprocess):
        self.records = records
        self.preprocess = preprocess
    def __len__(self):
        return len(self.records)
    def __getitem__(self, idx):
        m = self.records[idx]
        img = Image.open(m['frame_path']).convert('RGB')
        label = 1.0 if str(m['label']).strip().lower() == 'ai' else 0.0
        return self.preprocess(img), torch.tensor(label)


def get_preprocess():
   
    #  RAW PIXEL TENSORS ONLY (NO IMAGENET NORMALIZATION)
    
    return transforms.Compose([
        transforms.Resize(232, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor()  # leaves pixels in [0, 1] for SRM math
    ])


def clean_label_value(value):
    label = str(value).strip().lower()
    if label not in {"ai", "real"}:
        raise ValueError(f"Unexpected label value: {value!r}. Expected 'ai' or 'real'.")
    return label


def label_to_target(value):
    return 1 if clean_label_value(value) == "ai" else 0


def resolve_targets(metadata):
    if 'label' not in metadata.columns:
        if 'target' not in metadata.columns:
            raise ValueError("metadata needs a 'label' column or a verified 'target' column.")
        print("WARNING: No 'label' column found. Using existing 'target' column without label verification.")
        metadata['target'] = metadata['target'].astype(int)
        return metadata

    metadata['label'] = metadata['label'].apply(clean_label_value)
    derived_target = metadata['label'].apply(label_to_target).astype(int)

    if 'target' in metadata.columns:
        existing_target = pd.to_numeric(metadata['target'], errors='coerce')
        mismatches = existing_target.notna() & (existing_target.astype('Int64') != derived_target.astype('Int64'))
        if mismatches.any():
            print(f"\nWARNING: Existing target column disagrees with label for {int(mismatches.sum())} rows.")
            print("Rebuilding target from label.")

    metadata['target'] = derived_target

    mixed_label_videos = metadata.groupby('video_name')['target'].nunique()
    mixed_label_videos = mixed_label_videos[mixed_label_videos > 1]
    if len(mixed_label_videos) > 0:
        raise ValueError(f"Some video_name groups contain mixed labels. Examples: {mixed_label_videos.head(20).to_dict()}")

    print("\n--- Training Label Distribution ---")
    print(metadata['label'].value_counts())
    print("\n--- Training Video Label Distribution ---")
    print(metadata.groupby('video_name')['target'].first().value_counts().rename(index={0: 'real', 1: 'ai'}))
    return metadata


def finetune_noise_cnn(train_records, preprocess, device, fold_seed):
    torch.manual_seed(fold_seed)
    noise_cnn = NoiseStreamCNN(out_dim=128).to(device)
    head = nn.Linear(128, 1).to(device)
    params = list(noise_cnn.parameters()) + list(head.parameters())
    optimizer = optim.Adam(params, lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()

    loader = DataLoader(FrameDataset(train_records, preprocess),
                        batch_size=BATCH_TRAIN, shuffle=True, num_workers=0)

    for epoch in range(FINETUNE_EPOCHS):
        noise_cnn.train(); head.train()
        tot_loss, correct, total = 0.0, 0, 0
        for imgs, labels in loader:
            imgs = imgs.to(device); labels = labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            logits = head(noise_cnn(imgs))
            loss = criterion(logits, labels)
            loss.backward(); optimizer.step()
            tot_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item(); total += labels.size(0)
        print(f"     epoch {epoch+1}/{FINETUNE_EPOCHS}: loss={tot_loss/len(loader):.4f} acc={correct/total:.4f}")

    noise_cnn.eval()
    return noise_cnn


def extract_noise_features(noise_cnn, all_records, preprocess, device):
    loader = DataLoader(FrameDataset(all_records, preprocess),
                        batch_size=BATCH_EXTRACT, shuffle=False, num_workers=0)
    out = []
    with torch.no_grad():
        for imgs, _ in loader:
            out.append(noise_cnn(imgs.to(device)).cpu().numpy())
    return np.vstack(out)


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
    """Fresh Farneback flow stats from disk. Deterministic from pixels — no
    leakage from computing once outside the fold loop. Writes result to
    output_path for inspection only."""
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
    a fresh StandardScaler per inner fold. Returns (best_c, best_inner_score)."""
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
    start = time.time()
    print("--- TRAIN BRANCH B TRIPLE-STREAM (MobileNetV3 + Noise CNN + Farneback Flow, 5-FOLD CV) ---")
    print("--- (leakage-free inner CV: fresh scaler per inner fold) ---\n")

    try:
        semantic = np.load(SEMANTIC_FEATURES_PATH)   # [N, 960]
        metadata = pd.read_csv(METADATA_PATH)
        print(f"Loaded {len(semantic)} frames with {semantic.shape[1]}-dim semantic features")
    except FileNotFoundError:
        print("Error: Files not found. Run feature_extractor_b_mobilenet.py first.")
        return

    if len(semantic) != len(metadata):
        print(f"Error: Feature/metadata mismatch ({len(semantic)} vs {len(metadata)})")
        return
    if 'frame_path' not in metadata.columns:
        print("Error: metadata needs 'frame_path' column.")
        return

    try:
        metadata = resolve_targets(metadata)
    except ValueError as e:
        print(f"Error: {e}")
        return

    y_all   = metadata['target'].values
    groups  = metadata['video_name'].values
    records = metadata[['frame_path', 'filename', 'video_name', 'label']].to_dict('records')

    preprocess = get_preprocess()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type != 'cuda':
        print("WARNING: no CUDA — per-fold noise CNN training will be slow on CPU.")

    # --- Compute optical flow ONCE outside the fold loop (no leakage; pixel-deterministic) ---
    print("\nBuilding prev-frame map (forward diff for first frame of each video)...")
    prev_paths = build_prev_paths(metadata)

    print("Computing Farneback optical flow (FRESH, no cache load)...")
    t0_flow = time.time()
    flow_stats = compute_optical_flow_stats(records, prev_paths, FLOW_OUTPUT)
    flow_time = time.time() - t0_flow
    print(f"   Flow shape: {flow_stats.shape}, "
          f"mean_mag avg={flow_stats[:,0].mean():.3f}, max_mag avg={flow_stats[:,2].mean():.3f}")

    gkf = GroupKFold(n_splits=N_FOLDS)
    frame_accs, vid_accs, fold_Cs, fold_inner_scores = [], [], [], []
    frame_precs, frame_recs, frame_f1s = [], [], []
    vid_precs, vid_recs, vid_f1s = [], [], []
    frame_aps, frame_aucs, vid_aucs = [], [], []
    infer_times, n_test_vids_list = [], []
    fold_num = 0
    cv_start = time.time()

    for train_idx, test_idx in gkf.split(semantic, y_all, groups):
        fold_num += 1
        print(f"\n=== FOLD {fold_num}/{N_FOLDS} ===")
        print(f"Train: {len(train_idx)} frames / {len(np.unique(groups[train_idx]))} videos | "
              f"Test: {len(test_idx)} frames / {len(np.unique(groups[test_idx]))} videos")

        train_records = [records[i] for i in train_idx]
        test_records_fold = [records[i] for i in test_idx]

        # 1. Fresh noise CNN on this fold's train videos only
        print(f"  Training fresh noise CNN on {len(train_records)} train frames...")
        t0_noise_train = time.time()
        noise_cnn = finetune_noise_cnn(train_records, preprocess, device, fold_seed=42 + fold_num)
        noise_train_time = time.time() - t0_noise_train

        # 2. Extract noise features for train frames (for SVM training)
        print(f"  Extracting noise features for {len(train_records)} train frames...")
        t0_noise_extract = time.time()
        noise_train_feat = extract_noise_features(noise_cnn, train_records, preprocess, device)
        noise_extract_time = time.time() - t0_noise_extract

        # 3. Extract noise features for test frames — this is the CNN part of inference
        t0_noise_infer = time.time()
        noise_test_feat = extract_noise_features(noise_cnn, test_records_fold, preprocess, device)
        noise_infer_time = time.time() - t0_noise_infer

        del noise_cnn
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # 4. Build feature matrices: semantic (960) + noise (128) + flow (5) = 1093 dims
        X_tr_full = np.hstack([semantic[train_idx], noise_train_feat, flow_stats[train_idx]])
        X_te_full = np.hstack([semantic[test_idx],  noise_test_feat,  flow_stats[test_idx]])
        y_tr, y_te = y_all[train_idx], y_all[test_idx]
        groups_tr, groups_te = groups[train_idx], groups[test_idx]

        # 5. C selection on UNSCALED features (fresh scaler per inner fold)
        best_c, best_inner_score = select_C_inner(X_tr_full, y_tr, groups_tr)
        fold_Cs.append(best_c)
        fold_inner_scores.append(best_inner_score)

        # 6. Outer scaler — fit on train only, transform test
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_full)
        X_te = scaler.transform(X_te_full)

        # 7. Train LogReg on scaled features
        clf = LogisticRegression(C=best_c, random_state=RANDOM_STATE,
                                 class_weight='balanced', max_iter=2000)
        t0_train = time.time()
        clf.fit(X_tr, y_tr)
        train_time = time.time() - t0_train

        # 8. Inference: CNN on test frames already timed above; add SVM predict
        t0_svm_infer = time.time()
        y_pred = clf.predict(X_te)
        svm_infer_time = time.time() - t0_svm_infer
        infer_time = noise_infer_time + svm_infer_time  # full end-to-end inference

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
    print("5-FOLD CV RESULTS (MobileNetV3 + Noise CNN + Optical Flow)")
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
    print("\nFitting FINAL deployable model on ALL data (noise CNN + logreg)...")
    t0_final = time.time()

    t0_final_noise_train = time.time()
    final_noise_cnn = finetune_noise_cnn(records, preprocess, device, fold_seed=42)
    final_noise_train_time = time.time() - t0_final_noise_train

    t0_final_noise_extract = time.time()
    final_noise = extract_noise_features(final_noise_cnn, records, preprocess, device)
    final_noise_extract_time = time.time() - t0_final_noise_extract

    full_all = np.hstack([semantic, final_noise, flow_stats])

    best_c_final, final_inner_score = select_C_inner(full_all, y_all, groups)
    inner_str = f"{final_inner_score:.4f}" if not np.isnan(final_inner_score) else "n/a"
    print(f"   Final selected C: {best_c_final}  (inner cv = {inner_str})")

    scaler_final = StandardScaler()
    X_all = scaler_final.fit_transform(full_all)
    final_clf = LogisticRegression(C=best_c_final, random_state=RANDOM_STATE,
                                   class_weight='balanced', max_iter=2000)
    t0_final_svm = time.time()
    final_clf.fit(X_all, y_all)
    final_svm_time = time.time() - t0_final_svm
    final_train_time = time.time() - t0_final

    noise_path = MODEL_OUT.replace('.pkl', '_noisecnn.pt')
    torch.save(final_noise_cnn.state_dict(), noise_path)
    joblib.dump({
        'model': final_clf,
        'scaler': scaler_final,
        'noise_cnn_state': noise_path,
        'flow_output_for_inspection': FLOW_OUTPUT,
        'feature_layout': '960 MobileNetV3 + 128 noise CNN + 5 Farneback flow = 1093 dim',
        'cv_frame_accuracy_mean': float(np.mean(frame_accs)),
        'cv_frame_accuracy_std':  float(np.std(frame_accs)),
        'cv_video_accuracy_mean': float(np.mean(vid_accs)),
        'cv_video_accuracy_std':  float(np.std(vid_accs)),
        'cv_per_fold_C': fold_Cs,
        'cv_per_fold_inner_scores': fold_inner_scores,
    }, MODEL_OUT)
    print(f"Saved logreg+scaler to {MODEL_OUT}")
    print(f"Saved noise CNN weights to {noise_path}")

    # --- Model Statistics ---
    n_features           = full_all.shape[1]
    cnn_total_params     = sum(p.numel() for p in final_noise_cnn.parameters())
    cnn_trainable_params = sum(p.numel() for p in final_noise_cnn.parameters() if p.requires_grad)
    cnn_size_mb          = sum(p.numel() * p.element_size() for p in final_noise_cnn.parameters()) / (1024 ** 2)
    lr_params            = final_clf.coef_.size + final_clf.intercept_.size
    lr_size_kb           = (lr_params * 8) / 1024   # float64 coefficients
    lr_flops             = 2 * n_features + 1        # dot-product + bias per sample

    try:
        from thop import profile as _thop_profile
        _dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            _cnn_flops, _ = _thop_profile(final_noise_cnn, inputs=(_dummy_input,), verbose=False)
        cnn_flops_str = f"{_cnn_flops / 1e6:.2f} M"
    except ImportError:
        cnn_flops_str = "N/A (pip install thop)"

    # single-sample CNN inference (warm-up then timed loop)
    final_noise_cnn.eval()
    _dummy_img  = torch.randn(1, 3, 224, 224).to(device)
    _dummy_feat = np.zeros((1, n_features))
    with torch.no_grad():
        for _ in range(20):
            final_noise_cnn(_dummy_img)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    _t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(200):
            final_noise_cnn(_dummy_img)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    cnn_infer_ms = (time.perf_counter() - _t0) / 200 * 1000

    # single-sample LogReg inference
    for _ in range(20):
        final_clf.predict(_dummy_feat)
    _t0 = time.perf_counter()
    for _ in range(1000):
        final_clf.predict(_dummy_feat)
    lr_infer_ms = (time.perf_counter() - _t0) / 1000 * 1000

    print("\n" + "="*70)
    print("MODEL STATISTICS")
    print("="*70)
    infer_per_video = sum(infer_times) / sum(n_test_vids_list)
    print(f"Optical Flow Time:               {flow_time:.1f}s")
    print(f"5-Fold CV Training Time:         {cv_total_time:.1f}s")
    print(f"Final Model Training Time:       {final_train_time:.1f}s")
    print(f"Inference Time per Video:        {infer_per_video*1000:.2f}ms")
    print(f"Feature dimensions:                  {n_features:,}")
    print(f"Noise CNN — total params:            {cnn_total_params:,}")
    print(f"Noise CNN — trainable params:        {cnn_trainable_params:,}")
    print(f"Noise CNN — model size:              {cnn_size_mb:.4f} MB")
    print(f"Noise CNN — FLOPs (224x224 input):   {cnn_flops_str}")
    print(f"LogReg    — parameters:              {lr_params:,}")
    print(f"LogReg    — model size:              {lr_size_kb:.2f} KB")
    print(f"LogReg    — FLOPs/sample:            {lr_flops:,}")
    print(f"Single-sample inference (CNN):       {cnn_infer_ms:.4f} ms")
    print(f"Single-sample inference (LogReg):    {lr_infer_ms:.4f} ms")
    print(f"Single-sample inference (CNN+LogReg):{cnn_infer_ms + lr_infer_ms:.4f} ms")

    print(f"\nTotal script time: {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()