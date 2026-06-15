import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import torch

# =================================================================
# FIX 1: FORCE EXACT PRECISION FOR SRM NOISE EXTRACTION ON CUDA
# =================================================================
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
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, average_precision_score, roc_auc_score)
import time
import random
from tqdm import tqdm

# =============================================================================
# TRIPLE-STREAM TRAINER (Branch B) — MobileNetV3 + Noise CNN + Farneback Flow
# -----------------------------------------------------------------------------
# Random train/test split variant (GroupShuffleSplit, test_size=0.2).
# Streams per frame:
#   1. Cached 960-dim MobileNetV3-Large semantic features.
#   2. 128-dim noise features from a noise CNN (SRM + 3 conv).
#      Noise CNN is trained on the train split videos only.
#   3. 5-dim Farneback optical flow stats (mean_mag, std_mag, max_mag,
#      mean_cos_angle, mean_sin_angle) per frame.
# Final feature vector: 960 + 128 + 5 = 1093-dim.
#
# CHANGE vs train_branch_b_triplestream_noise_flow.py:
#   * 5-fold GroupKFold cross-validation replaced with a single random
#     GroupShuffleSplit (test_size=0.2). All other logic is identical.
#
# CACHE POLICY:
#   * Optical flow is recomputed FRESHLY on every run (no cache load).
#   * Semantic features are loaded from the cache produced by the upstream
#     MobileNetV3 extractor.
#   * Noise features are recomputed once on the train split.
#
# Inner CV: LEAKAGE-FREE. select_C_inner receives UNSCALED features and fits
# a fresh StandardScaler per inner fold.
# =============================================================================

SEMANTIC_FEATURES_PATH = "branch_b_features_dualstream_fa.npy"
METADATA_PATH          = "branch_b_metadata_dualstream_fa.csv"
FLOW_OUTPUT            = "branch_b_flow_farneback_triplestream_randomsplit.npy"
MODEL_OUT              = "model_branch_b_triplestream_noise_flow_randomsplit.pkl"

# Noise CNN fine-tune config
BATCH_TRAIN     = 64
BATCH_EXTRACT   = 32
FINETUNE_EPOCHS = 5
FINETUNE_LR     = 5e-4
WEIGHT_DECAY    = 5e-4

# split / logreg config
TEST_SIZE    = 0.2
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
    # =================================================================
    # FIX 2: RAW PIXEL TENSORS ONLY (NO IMAGENET NORMALIZATION)
    # =================================================================
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
    print("--- TRAIN BRANCH B TRIPLE-STREAM (MobileNetV3 + Noise CNN + Farneback Flow, RANDOM SPLIT, NO PCA) ---")
    print("--- (leakage-free inner CV: fresh scaler per inner fold) ---\n")

    try:
        semantic = np.load(SEMANTIC_FEATURES_PATH)   # [N, 960]
        metadata = pd.read_csv(METADATA_PATH)
        print(f"Loaded {len(semantic)} frames with {semantic.shape[1]}-dim semantic features")
    except FileNotFoundError:
        print("Error: Files not found. Run branch_b_dualstream_fa_mobilenetv3_clean.py first.")
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
        print("WARNING: no CUDA — noise CNN training will be slow on CPU.")

    # --- Compute optical flow ONCE (no leakage; pixel-deterministic) ---
    print("\nBuilding prev-frame map (forward diff for first frame of each video)...")
    prev_paths = build_prev_paths(metadata)

    print("Computing Farneback optical flow (FRESH, no cache load)...")
    t0_flow = time.time()
    flow_stats = compute_optical_flow_stats(records, prev_paths, FLOW_OUTPUT)
    flow_time = time.time() - t0_flow
    print(f"   Flow shape: {flow_stats.shape}, "
          f"mean_mag avg={flow_stats[:,0].mean():.3f}, max_mag avg={flow_stats[:,2].mean():.3f}")

    print(f"\nRunning random split (test_size={TEST_SIZE}) evaluation (NO PCA)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(semantic, y_all, groups))

    print(f"Train: {len(train_idx)} frames / {len(np.unique(groups[train_idx]))} videos | "
          f"Test: {len(test_idx)} frames / {len(np.unique(groups[test_idx]))} videos")

    # 1. Train noise CNN on train videos only
    train_records = [records[i] for i in train_idx]
    print(f"  Training noise CNN on {len(train_records)} train frames...")
    t0_noise_train = time.time()
    noise_cnn = finetune_noise_cnn(train_records, preprocess, device, fold_seed=43)
    noise_train_time = time.time() - t0_noise_train

    # 2. Extract noise features for ALL frames with this noise CNN
    print(f"  Extracting noise features for all {len(records)} frames...")
    t0_noise_extract = time.time()
    noise = extract_noise_features(noise_cnn, records, preprocess, device)
    noise_extract_time = time.time() - t0_noise_extract
    del noise_cnn
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # 3. Combine streams: semantic (960) + noise (128) + flow (5) = 1093 dims
    full = np.hstack([semantic, noise, flow_stats])

    # 4. Split into train/test
    X_tr_full, X_te_full = full[train_idx], full[test_idx]
    y_tr, y_te = y_all[train_idx], y_all[test_idx]
    groups_tr, groups_te = groups[train_idx], groups[test_idx]

    # 5. C selection on UNSCALED features (fresh scaler per inner fold)
    best_c, best_inner_score = select_C_inner(X_tr_full, y_tr, groups_tr)

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

    inner_str = f"{best_inner_score:.4f}" if not np.isnan(best_inner_score) else "n/a"
    print(f"  C={best_c}  inner_cv={inner_str}  frame={f_acc:.4f}  video={v_acc:.4f}")
    print(f"  frame prec={f_prec:.4f} rec={f_rec:.4f} f1={f_f1:.4f} AP={f_ap:.4f} AUC={f_auc:.4f}")
    print(f"  video prec={v_prec:.4f} rec={v_rec:.4f} f1={v_f1:.4f}")
    total_train_time = flow_time + noise_train_time + noise_extract_time + train_time
    print(f"  train={train_time:.3f}s  infer={infer_time:.4f}s  test={test_time:.4f}s")

    print("\n" + "="*70)
    print("RANDOM SPLIT RESULTS (MobileNetV3 + Noise CNN + Optical Flow)")
    print("="*70)
    print(f"Selected C:              {best_c}")
    print(f"Inner CV score:          {inner_str}")
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
    print(f"\nOptical Flow Time:       {flow_time:.1f}s")
    print(f"Noise CNN Training Time: {noise_train_time:.1f}s")
    print(f"Noise Extraction Time:   {noise_extract_time:.1f}s")
    print(f"SVM Training Time:       {train_time:.3f}s")
    print(f"Total Training Time:     {total_train_time:.1f}s")
    print(f"Inference Time:          {infer_time:.4f}s")
    print(f"Testing Time:            {test_time:.4f}s")

    print(f"\nTotal script time: {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
