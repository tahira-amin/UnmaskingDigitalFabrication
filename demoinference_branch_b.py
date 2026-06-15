import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Force exact precision for SRM noise extraction (must match training)
import torch
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import os
import time
import numpy as np
import pandas as pd
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, average_precision_score,
                              confusion_matrix)
import joblib

# =============================================================================
# DEMO INFERENCE — Branch B (MobileNetV3 + Noise CNN + Farneback Flow)
# -----------------------------------------------------------------------------
# Reads demo_routing.csv, loads Branch B videos from demo_frames/, extracts
# the same 1093-dim feature vector used during training, and runs the saved
# Branch B model. Ground truth = AI (1) for all Sora videos.
# =============================================================================

ROUTING_CSV    = "demo_routing.csv"
FRAMES_ROOT    = "demo_frames"
MODEL_PKL      = "model_branch_b_triplestream_noise_flow.pkl"
NOISE_CNN_PT   = "model_branch_b_triplestream_noise_flow_noisecnn.pt"

GROUND_TRUTH   = 1        # All Sora videos are AI-generated
BATCH_SIZE     = 32
FLOW_RESIZE    = 224      # Must match training


# ── NOISE CNN (SRM + 3-conv → 128-dim) ───────────────────────────────────────

SRM_KERNELS = np.array([
    [[ 0, 0, 0, 0, 0],[ 0, 0, 0, 0, 0],[ 0, 1,-2, 1, 0],[ 0, 0, 0, 0, 0],[ 0, 0, 0, 0, 0]],
    [[ 0, 0, 0, 0, 0],[ 0,-1, 2,-1, 0],[ 0, 2,-4, 2, 0],[ 0,-1, 2,-1, 0],[ 0, 0, 0, 0, 0]],
    [[-1, 2,-2, 2,-1],[ 2,-6, 8,-6, 2],[-2, 8,-12,8,-2],[ 2,-6, 8,-6, 2],[-1, 2,-2, 2,-1]],
], dtype=np.float32)
SRM_NORMS = np.array([2.0, 4.0, 12.0], dtype=np.float32)


class NoiseStreamCNN(nn.Module):
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
        self.conv1 = nn.Conv2d(9,  32, 3, stride=2, padding=1, bias=False)
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


# ── MOBILENETV3 EXTRACTOR ─────────────────────────────────────────────────────

class MobileNetExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        w = models.MobileNet_V3_Large_Weights.DEFAULT
        m = models.mobilenet_v3_large(weights=w)
        self.features   = m.features
        self.avgpool    = m.avgpool
        self.flatten    = nn.Flatten()
        self.preprocess = w.transforms()
    def forward(self, x):
        return self.flatten(self.avgpool(self.features(x)))


# ── NOISE PREPROCESSING (raw pixels, no ImageNet norm — must match training) ──

def get_noise_preprocess():
    return transforms.Compose([
        transforms.Resize(232, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),   # [0, 1] only — SRM needs raw pixel values
    ])


# ── FRAME COLLECTION ──────────────────────────────────────────────────────────

def collect_frames(video_names):
    records, missing = [], []
    for vn in sorted(video_names):
        folder = os.path.join(FRAMES_ROOT, vn)
        if not os.path.isdir(folder):
            missing.append(vn)
            continue
        files = sorted(f for f in os.listdir(folder)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        for f in files:
            records.append({
                'video_name': vn,
                'filename':   f,
                'frame_path': os.path.join(folder, f),
            })
    if missing:
        print(f"WARNING: {len(missing)} videos missing from {FRAMES_ROOT}/: {missing}")
    return records


# ── BATCHED MOBILENETV3 EXTRACTION ────────────────────────────────────────────

def extract_mobilenet_features(records, extractor, device):
    n = len(records)
    feats, ok_idx = [], []
    skipped = 0
    for i in tqdm(range(0, n, BATCH_SIZE),
                  total=(n + BATCH_SIZE - 1) // BATCH_SIZE,
                  desc="MobileNetV3"):
        batch_meta = records[i:i+BATCH_SIZE]
        tensors, local_ok = [], []
        for j, m in enumerate(batch_meta):
            try:
                img = Image.open(m['frame_path']).convert('RGB')
                tensors.append(extractor.preprocess(img))
                local_ok.append(i + j)
            except Exception:
                skipped += 1
        if not tensors:
            continue
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            out = extractor(batch).cpu().numpy()
        feats.append(out)
        ok_idx.extend(local_ok)
    if skipped:
        print(f"  [MobileNetV3] skipped {skipped} frames")
    return np.vstack(feats) if feats else np.empty((0, 0)), ok_idx


# ── NOISE FEATURE EXTRACTION ──────────────────────────────────────────────────

class _FrameDataset(Dataset):
    def __init__(self, records, preprocess):
        self.records    = records
        self.preprocess = preprocess
    def __len__(self):
        return len(self.records)
    def __getitem__(self, idx):
        img = Image.open(self.records[idx]['frame_path']).convert('RGB')
        return self.preprocess(img), torch.tensor(0.0)   # label unused at inference


def extract_noise_features(noise_cnn, records, preprocess, device):
    loader = DataLoader(_FrameDataset(records, preprocess),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    out = []
    noise_cnn.eval()
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="Noise CNN"):
            out.append(noise_cnn(imgs.to(device)).cpu().numpy())
    return np.vstack(out) if out else np.empty((0, 0))


# ── FARNEBACK OPTICAL FLOW ────────────────────────────────────────────────────

def build_prev_paths(records):
    meta = pd.DataFrame(records).copy()
    meta['frame_num'] = meta['filename'].str.extract(r'(\d+)').astype(int)
    prev_paths = [None] * len(meta)
    for _, group in meta.groupby('video_name'):
        sg    = group.sort_values('frame_num')
        idxs  = sg.index.tolist()
        paths = sg['frame_path'].tolist()
        n = len(idxs)
        for i, orig in enumerate(idxs):
            prev_paths[orig] = paths[1] if (i == 0 and n > 1) else paths[max(i-1, 0)]
    return prev_paths


def compute_flow_stats(records, prev_paths):
    stats = np.zeros((len(records), 5), dtype=np.float32)
    for i, m in enumerate(tqdm(records, desc="Farneback flow")):
        try:
            curr = cv2.imread(m['frame_path'], cv2.IMREAD_GRAYSCALE)
            prev = cv2.imread(prev_paths[i],   cv2.IMREAD_GRAYSCALE)
            if curr is None or prev is None:
                continue
            if curr.shape != prev.shape:
                prev = cv2.resize(prev, (curr.shape[1], curr.shape[0]))
            cr = cv2.resize(curr, (FLOW_RESIZE, FLOW_RESIZE))
            pr = cv2.resize(prev, (FLOW_RESIZE, FLOW_RESIZE))
            flow = cv2.calcOpticalFlowFarneback(
                pr, cr, None,
                pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                poly_n=5, poly_sigma=1.2, flags=0
            )
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            stats[i] = [np.mean(mag), np.std(mag), np.max(mag),
                        np.mean(np.cos(ang)), np.mean(np.sin(ang))]
        except Exception as e:
            print(f"  flow skip {m['frame_path']}: {e}")
    return stats


# ── VIDEO-LEVEL AGGREGATION ───────────────────────────────────────────────────

def video_level(video_names, y_pred, y_proba):
    df = pd.DataFrame({'video': video_names, 'pred': y_pred, 'proba': y_proba})
    return df.groupby('video').agg(
        pred_video =('pred',  lambda x: int(x.mode()[0])),
        mean_proba =('proba', 'mean'),
        n_frames   =('pred',  'count'),
        n_ai_frames=('pred',  'sum'),
    ).reset_index()


# ── METRICS REPORT ────────────────────────────────────────────────────────────

def report_metrics(y_true_frame, y_pred_frame, y_proba_frame,
                   y_true_vid,   y_pred_vid,   y_proba_vid):
    print("\n" + "="*60)
    print("  FRAME-LEVEL METRICS")
    print("="*60)
    print(f"Total frames:   {len(y_true_frame)}")
    print(f"Accuracy:       {accuracy_score(y_true_frame, y_pred_frame):.4f}")
    print(f"Precision (AI): {precision_score(y_true_frame, y_pred_frame, zero_division=0):.4f}")
    print(f"Recall    (AI): {recall_score(y_true_frame, y_pred_frame, zero_division=0):.4f}")
    print(f"F1        (AI): {f1_score(y_true_frame, y_pred_frame, zero_division=0):.4f}")
    if len(np.unique(y_true_frame)) > 1:
        print(f"ROC-AUC:        {roc_auc_score(y_true_frame, y_proba_frame):.4f}")
        print(f"Avg Precision:  {average_precision_score(y_true_frame, y_proba_frame):.4f}")
    cm = confusion_matrix(y_true_frame, y_pred_frame, labels=[0, 1])
    print(f"\nConfusion Matrix (frame-level):")
    print(f"  True AI  -> Pred AI:   {cm[1][1]}")
    print(f"  True AI  -> Pred Real: {cm[1][0]}")

    print("\n" + "="*60)
    print("  VIDEO-LEVEL METRICS")
    print("="*60)
    print(f"Total videos:   {len(y_true_vid)}")
    print(f"Accuracy:       {accuracy_score(y_true_vid, y_pred_vid):.4f}")
    print(f"Precision (AI): {precision_score(y_true_vid, y_pred_vid, zero_division=0):.4f}")
    print(f"Recall    (AI): {recall_score(y_true_vid, y_pred_vid, zero_division=0):.4f}")
    print(f"F1        (AI): {f1_score(y_true_vid, y_pred_vid, zero_division=0):.4f}")
    if len(np.unique(y_true_vid)) > 1:
        print(f"ROC-AUC:        {roc_auc_score(y_true_vid, y_proba_vid):.4f}")
        print(f"Avg Precision:  {average_precision_score(y_true_vid, y_proba_vid):.4f}")
    correct   = int((np.array(y_pred_vid) == np.array(y_true_vid)).sum())
    incorrect = len(y_true_vid) - correct
    print(f"\nCorrectly identified as AI: {correct}/{len(y_true_vid)}")
    print(f"Misclassified as Real:      {incorrect}/{len(y_true_vid)}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    start = time.time()
    print("--- DEMO INFERENCE: Branch B (MobileNetV3 + Noise CNN + Farneback Flow) ---\n")

    # 1. Load routing CSV and filter Branch B
    if not os.path.exists(ROUTING_CSV):
        print(f"Error: {ROUTING_CSV} not found. Run demorouter.py first.")
        return
    routing  = pd.read_csv(ROUTING_CSV)
    branch_b = routing[routing['assigned_branch'] == 'Branch B']
    video_names = branch_b['video_name'].tolist()
    print(f"Branch B videos from routing: {len(video_names)}")
    if not video_names:
        print("No Branch B videos found in routing CSV.")
        return
    for v in video_names:
        print(f"  {v}")

    # 2. Collect frames
    print(f"\nCollecting frames from {FRAMES_ROOT}/...")
    records = collect_frames(video_names)
    print(f"Found {len(records)} frames across {len(set(r['video_name'] for r in records))} videos.")
    if not records:
        return

    # 3. Load saved model bundle
    for path in (MODEL_PKL, NOISE_CNN_PT):
        if not os.path.exists(path):
            print(f"Error: {path} not found. Run the Branch B trainer first.")
            return

    bundle  = joblib.load(MODEL_PKL)
    clf     = bundle['model']
    scaler  = bundle['scaler']
    layout  = bundle.get('feature_layout', '960 + 128 + 5 = 1093 dim')
    cv_acc  = bundle.get('cv_frame_accuracy_mean', float('nan'))
    cv_vid  = bundle.get('cv_video_accuracy_mean',  float('nan'))
    print(f"\nLoaded model : {MODEL_PKL}")
    print(f"  Feature layout : {layout}")
    print(f"  CV frame acc   : {cv_acc:.4f}")
    print(f"  CV video acc   : {cv_vid:.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 4. Load noise CNN weights
    noise_cnn = NoiseStreamCNN(out_dim=128).to(device)
    noise_cnn.load_state_dict(torch.load(NOISE_CNN_PT, map_location=device))
    noise_cnn.eval()
    print(f"Loaded noise CNN weights: {NOISE_CNN_PT}")

    # 5. MobileNetV3-Large features (960-dim)
    print("\n--- Extracting MobileNetV3-Large features (960-dim) ---")
    mob_extractor = MobileNetExtractor().to(device).eval()
    mob_feats, mob_ok = extract_mobilenet_features(records, mob_extractor, device)
    del mob_extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 6. Noise CNN features (128-dim) — raw pixel preprocessing, no ImageNet norm
    print("\n--- Extracting Noise CNN features (128-dim) ---")
    noise_preprocess = get_noise_preprocess()
    # Align records to those that loaded successfully in MobileNet step
    aligned_records = [records[i] for i in mob_ok]
    noise_feats = extract_noise_features(noise_cnn, aligned_records, noise_preprocess, device)
    del noise_cnn
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 7. Farneback optical flow (5-dim)
    print("\n--- Computing Farneback optical flow (5-dim) ---")
    prev_paths = build_prev_paths(aligned_records)
    flow_stats = compute_flow_stats(aligned_records, prev_paths)
    print(f"  Flow shape: {flow_stats.shape}, "
          f"mean_mag avg={flow_stats[:,0].mean():.3f}, max_mag avg={flow_stats[:,2].mean():.3f}")

    # 8. Combine: 960 + 128 + 5 = 1093
    X = np.hstack([mob_feats, noise_feats, flow_stats])
    print(f"\nCombined feature matrix: {X.shape}")

    # 9. Scale + predict
    X_scaled = scaler.transform(X)
    y_pred   = clf.predict(X_scaled)
    y_proba  = clf.predict_proba(X_scaled)[:, 1]

    # 10. Ground truth (all Sora = AI)
    y_true_frame = np.full(len(y_pred), GROUND_TRUTH, dtype=int)
    video_arr    = np.array([r['video_name'] for r in aligned_records])

    # 11. Video-level aggregation
    vid_df = video_level(video_arr, y_pred, y_proba)
    vid_df['ground_truth'] = GROUND_TRUTH
    y_true_vid  = vid_df['ground_truth'].values
    y_pred_vid  = vid_df['pred_video'].values
    y_proba_vid = vid_df['mean_proba'].values

    # 12. Metrics
    report_metrics(y_true_frame, y_pred_frame=y_pred, y_proba_frame=y_proba,
                   y_true_vid=y_true_vid, y_pred_vid=y_pred_vid, y_proba_vid=y_proba_vid)

    # 13. Per-video table
    print("\n" + "="*60)
    print("  PER-VIDEO RESULTS")
    print("="*60)
    vid_df['prediction'] = vid_df['pred_video'].apply(lambda x: 'AI' if x == 1 else 'Real')
    vid_df['correct']    = (vid_df['pred_video'] == GROUND_TRUTH).map({True: 'YES', False: 'NO'})
    print(vid_df[['video', 'prediction', 'mean_proba', 'n_frames', 'n_ai_frames', 'correct']]
          .sort_values('mean_proba', ascending=False)
          .to_string(index=False))

    elapsed = time.time() - start
    print(f"\nDemo inference completed in {elapsed:.1f} seconds.")


if __name__ == "__main__":
    main()
