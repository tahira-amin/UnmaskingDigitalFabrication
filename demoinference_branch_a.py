import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import os
import time
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, average_precision_score,
                              confusion_matrix)
import joblib

# =============================================================================
# DEMO INFERENCE — Branch A (EffNet + MobileNetV3 + Farneback Flow)
# -----------------------------------------------------------------------------
# Reads demo_routing.csv, loads Branch A videos from demo_frames/, extracts
# the same 2245-dim feature vector used during training, and runs the saved
# Branch A model. Ground truth = AI (1) for all Sora videos.
# =============================================================================

ROUTING_CSV  = "demo_routing.csv"
FRAMES_ROOT  = "demo_frames"
MODEL_PATH   = "model_branch_a_triplestream_farneback.pkl"

GROUND_TRUTH = 1          # All Sora videos are AI-generated
BATCH_SIZE   = 32
FLOW_RESIZE  = 224        # Must match training


# ── FEATURE EXTRACTORS ────────────────────────────────────────────────────────

class EffNetExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        w = models.EfficientNet_B0_Weights.DEFAULT
        m = models.efficientnet_b0(weights=w)
        self.features = m.features
        self.avgpool  = m.avgpool
        self.flatten  = nn.Flatten()
        self.preprocess = w.transforms()
    def forward(self, x):
        return self.flatten(self.avgpool(self.features(x)))


class MobileNetExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        w = models.MobileNet_V3_Large_Weights.DEFAULT
        m = models.mobilenet_v3_large(weights=w)
        self.features = m.features
        self.avgpool  = m.avgpool
        self.flatten  = nn.Flatten()
        self.preprocess = w.transforms()
    def forward(self, x):
        return self.flatten(self.avgpool(self.features(x)))


def load_extractors():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    eff  = EffNetExtractor().to(device).eval()
    mob  = MobileNetExtractor().to(device).eval()
    return eff, mob, device


# ── FRAME COLLECTION ─────────────────────────────────────────────────────────

def collect_frames(video_names):
    records = []
    missing = []
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


# ── CNN FEATURE EXTRACTION (batched) ─────────────────────────────────────────

def extract_cnn_features(records, extractor, device, label):
    n = len(records)
    n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE
    feats, ok_idx = [], []
    skipped = 0
    for i in tqdm(range(0, n, BATCH_SIZE), total=n_batches, desc=label):
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
        print(f"  [{label}] skipped {skipped} frames")
    return np.vstack(feats) if feats else np.empty((0, 0)), ok_idx


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
    g = df.groupby('video').agg(
        pred_video =('pred',  lambda x: int(x.mode()[0])),
        mean_proba =('proba', 'mean'),
        n_frames   =('pred',  'count'),
        n_ai_frames=('pred',  'sum'),
    ).reset_index()
    return g


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
    cm = confusion_matrix(y_true_frame, y_pred_frame)
    print(f"\nConfusion Matrix (frame-level):")
    print(f"  True AI  -> Pred AI:   {cm[0][0] if cm.shape == (1,1) else cm[1][1]}")
    print(f"  True AI  -> Pred Real: {0 if cm.shape == (1,1) else cm[1][0]}")

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
    print("--- DEMO INFERENCE: Branch A (EffNet + MobileNetV3 + Farneback Flow) ---\n")

    # 1. Load routing CSV and filter Branch A
    if not os.path.exists(ROUTING_CSV):
        print(f"Error: {ROUTING_CSV} not found. Run demorouter.py first.")
        return
    routing = pd.read_csv(ROUTING_CSV)
    branch_a = routing[routing['assigned_branch'] == 'Branch A']
    video_names = branch_a['video_name'].tolist()
    print(f"Branch A videos from routing: {len(video_names)}")
    if not video_names:
        print("No Branch A videos found in routing CSV.")
        return

    # 2. Collect frames
    print(f"\nCollecting frames from {FRAMES_ROOT}/...")
    records = collect_frames(video_names)
    print(f"Found {len(records)} frames across {len(set(r['video_name'] for r in records))} videos.")
    if not records:
        return

    # 3. Load saved model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found. Run the Branch A trainer first.")
        return
    bundle  = joblib.load(MODEL_PATH)
    clf     = bundle['model']
    scaler  = bundle['scaler']
    layout  = bundle.get('feature_layout', '1280 + 960 + 5 = 2245 dim')
    cv_acc  = bundle.get('cv_frame_accuracy_mean', float('nan'))
    cv_vid  = bundle.get('cv_video_accuracy_mean',  float('nan'))
    print(f"\nLoaded model: {MODEL_PATH}")
    print(f"  Feature layout : {layout}")
    print(f"  CV frame acc   : {cv_acc:.4f}")
    print(f"  CV video acc   : {cv_vid:.4f}")

    # 4. EfficientNet-B0 features (1280-dim)
    print("\n--- Extracting EfficientNet-B0 features (1280-dim) ---")
    eff, mob, device = load_extractors()
    eff_feats, eff_ok = extract_cnn_features(records, eff, device, "EfficientNet-B0")
    del eff
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5. MobileNetV3-Large features (960-dim)
    print("\n--- Extracting MobileNetV3-Large features (960-dim) ---")
    mob_feats, mob_ok = extract_cnn_features(records, mob, device, "MobileNetV3")
    del mob
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Align both streams to the intersection of successfully loaded frames
    ok_idx = sorted(set(eff_ok) & set(mob_ok))
    if len(ok_idx) != len(records):
        print(f"  Alignment: kept {len(ok_idx)}/{len(records)} frames (both CNNs succeeded).")

    eff_map = {idx: i for i, idx in enumerate(eff_ok)}
    mob_map = {idx: i for i, idx in enumerate(mob_ok)}
    aligned_records  = [records[i]    for i in ok_idx]
    eff_aligned      = eff_feats[[eff_map[i] for i in ok_idx]]
    mob_aligned      = mob_feats[[mob_map[i] for i in ok_idx]]

    # 6. Farneback optical flow (5-dim)
    print("\n--- Computing Farneback optical flow (5-dim) ---")
    prev_paths   = build_prev_paths(aligned_records)
    flow_stats   = compute_flow_stats(aligned_records, prev_paths)
    print(f"  Flow shape: {flow_stats.shape}, "
          f"mean_mag avg={flow_stats[:,0].mean():.3f}, max_mag avg={flow_stats[:,2].mean():.3f}")

    # 7. Combine: 1280 + 960 + 5 = 2245
    X = np.hstack([eff_aligned, mob_aligned, flow_stats])
    print(f"\nCombined feature matrix: {X.shape}")

    # 8. Scale + predict
    X_scaled  = scaler.transform(X)
    y_pred    = clf.predict(X_scaled)
    y_proba   = clf.predict_proba(X_scaled)[:, 1]

    # 9. Ground truth (all Sora = AI)
    y_true_frame = np.full(len(y_pred), GROUND_TRUTH, dtype=int)
    video_arr    = np.array([r['video_name'] for r in aligned_records])

    # 10. Video-level aggregation
    vid_df = video_level(video_arr, y_pred, y_proba)
    vid_df['ground_truth'] = GROUND_TRUTH
    y_true_vid = vid_df['ground_truth'].values
    y_pred_vid = vid_df['pred_video'].values
    y_proba_vid = vid_df['mean_proba'].values

    # 11. Metrics
    report_metrics(y_true_frame, y_pred_frame=y_pred, y_proba_frame=y_proba,
                   y_true_vid=y_true_vid, y_pred_vid=y_pred_vid, y_proba_vid=y_proba_vid)

    # 12. Per-video table
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
