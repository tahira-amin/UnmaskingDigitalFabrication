"""
demoinference.py  —  single-video inference
Usage:
    python demoinference.py <path/to/video.mp4>  |  <video_name>

Reads demo_routing.csv to find the video's assigned branch, then runs the
matching saved model with branch-specific feature extraction:

  Branch A  ->  EfficientNet-B0 (1280) + MobileNetV3 (960) + Farneback (5) = 2245-dim
               model_branch_a_triplestream_farneback.pkl

  Branch B  ->  MobileNetV3 (960) + Noise CNN (128) + Farneback (5) = 1093-dim
               model_branch_b_triplestream_noise_flow.pkl
               model_branch_b_triplestream_noise_flow_noisecnn.pt

Outputs per-frame predictions and the final video-level prediction.
"""
import sys

# SRM precision fix (must be set before any other torch import)
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
import joblib

ROUTING_CSV  = "demo_routing.csv"
FRAMES_ROOT  = "demo_frames"

MODEL_A_PKL    = "model_branch_a_triplestream_farneback.pkl"
MODEL_B_PKL    = "model_branch_b_triplestream_noise_flow.pkl"
MODEL_B_NOISE  = "model_branch_b_triplestream_noise_flow_noisecnn.pt"

BATCH_SIZE  = 32
FLOW_RESIZE = 224


# ── BRANCH A: EfficientNet-B0 extractor (1280-dim) ───────────────────────────

class EffNetExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        w = models.EfficientNet_B0_Weights.DEFAULT
        m = models.efficientnet_b0(weights=w)
        self.features   = m.features
        self.avgpool    = m.avgpool
        self.flatten    = nn.Flatten()
        self.preprocess = w.transforms()
    def forward(self, x):
        return self.flatten(self.avgpool(self.features(x)))


# ── BRANCH A & B: MobileNetV3-Large extractor (960-dim) ──────────────────────

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


# ── BRANCH B: Noise CNN (SRM + 3-conv -> 128-dim) ─────────────────────────────

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

def noise_preprocess():
    # Raw pixels [0,1] — NO ImageNet normalisation; SRM needs unmodified values
    return transforms.Compose([
        transforms.Resize(232, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])


# ── SHARED: FRAME COLLECTION ──────────────────────────────────────────────────

def collect_frames(video_name):
    folder = os.path.join(FRAMES_ROOT, video_name)
    if not os.path.isdir(folder):
        return []
    return sorted(
        [os.path.join(folder, f) for f in os.listdir(folder)
         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    )


# ── SHARED: BATCHED CNN EXTRACTION ────────────────────────────────────────────

def extract_cnn(frame_paths, extractor, device, label):
    feats = []
    for i in tqdm(range(0, len(frame_paths), BATCH_SIZE),
                  total=(len(frame_paths) + BATCH_SIZE - 1) // BATCH_SIZE,
                  desc=label, leave=False, file=sys.stderr):
        batch_paths = frame_paths[i:i+BATCH_SIZE]
        tensors = []
        for p in batch_paths:
            img = Image.open(p).convert('RGB')
            tensors.append(extractor.preprocess(img))
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            feats.append(extractor(batch).cpu().numpy())
    return np.vstack(feats)


# ── SHARED: NOISE CNN EXTRACTION ─────────────────────────────────────────────

class _FrameList(Dataset):
    def __init__(self, paths, preprocess):
        self.paths      = paths
        self.preprocess = preprocess
    def __len__(self):  return len(self.paths)
    def __getitem__(self, i):
        return self.preprocess(Image.open(self.paths[i]).convert('RGB')), 0.0

def extract_noise(frame_paths, noise_cnn, device):
    loader = DataLoader(_FrameList(frame_paths, noise_preprocess()),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    feats = []
    noise_cnn.eval()
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="Noise CNN", leave=False, file=sys.stderr):
            feats.append(noise_cnn(imgs.to(device)).cpu().numpy())
    return np.vstack(feats)


# ── SHARED: FARNEBACK OPTICAL FLOW ───────────────────────────────────────────

def compute_flow(frame_paths):
    n = len(frame_paths)
    stats = np.zeros((n, 5), dtype=np.float32)
    for i in tqdm(range(n), desc="Farneback flow", leave=False, file=sys.stderr):
        prev_path = frame_paths[i+1] if i == 0 and n > 1 else frame_paths[max(i-1, 0)]
        try:
            curr = cv2.imread(frame_paths[i], cv2.IMREAD_GRAYSCALE)
            prev = cv2.imread(prev_path,      cv2.IMREAD_GRAYSCALE)
            if curr is None or prev is None: continue
            if curr.shape != prev.shape:
                prev = cv2.resize(prev, (curr.shape[1], curr.shape[0]))
            cr = cv2.resize(curr, (FLOW_RESIZE, FLOW_RESIZE))
            pr = cv2.resize(prev, (FLOW_RESIZE, FLOW_RESIZE))
            flow = cv2.calcOpticalFlowFarneback(
                pr, cr, None,
                pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                poly_n=5, poly_sigma=1.2, flags=0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            stats[i] = [np.mean(mag), np.std(mag), np.max(mag),
                        np.mean(np.cos(ang)), np.mean(np.sin(ang))]
        except Exception as e:
            print(f"  flow skip frame {i}: {e}")
    return stats


# ── BRANCH A INFERENCE ────────────────────────────────────────────────────────

def infer_branch_a(frame_paths, device):
    bundle = joblib.load(MODEL_A_PKL)
    clf, scaler = bundle['model'], bundle['scaler']
    layout      = bundle.get('feature_layout', '1280+960+5=2245')
    cv_frame    = bundle.get('cv_frame_accuracy_mean', float('nan'))
    cv_video    = bundle.get('cv_video_accuracy_mean',  float('nan'))
    print(f"  Model      : {MODEL_A_PKL}")
    print(f"  Layout     : {layout}")
    print(f"  CV frame acc: {cv_frame:.4f}  |  CV video acc: {cv_video:.4f}")

    print("  Extracting EfficientNet-B0 features (1280-dim)...")
    eff = EffNetExtractor().to(device).eval()
    eff_feats = extract_cnn(frame_paths, eff, device, "EfficientNet-B0")
    del eff; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("  Extracting MobileNetV3-Large features (960-dim)...")
    mob = MobileNetExtractor().to(device).eval()
    mob_feats = extract_cnn(frame_paths, mob, device, "MobileNetV3")
    del mob; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("  Computing Farneback optical flow (5-dim)...")
    flow = compute_flow(frame_paths)

    X = np.hstack([eff_feats, mob_feats, flow])
    print(f"  Feature matrix: {X.shape}")
    X_sc     = scaler.transform(X)
    y_pred   = clf.predict(X_sc)
    y_proba  = clf.predict_proba(X_sc)[:, 1]
    return y_pred, y_proba


# ── BRANCH B INFERENCE ────────────────────────────────────────────────────────

def infer_branch_b(frame_paths, device):
    for path in (MODEL_B_PKL, MODEL_B_NOISE):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
    bundle = joblib.load(MODEL_B_PKL)
    clf, scaler = bundle['model'], bundle['scaler']
    layout      = bundle.get('feature_layout', '960+128+5=1093')
    cv_frame    = bundle.get('cv_frame_accuracy_mean', float('nan'))
    cv_video    = bundle.get('cv_video_accuracy_mean',  float('nan'))
    print(f"  Model      : {MODEL_B_PKL}")
    print(f"  Layout     : {layout}")
    print(f"  CV frame acc: {cv_frame:.4f}  |  CV video acc: {cv_video:.4f}")

    print("  Extracting MobileNetV3-Large features (960-dim)...")
    mob = MobileNetExtractor().to(device).eval()
    mob_feats = extract_cnn(frame_paths, mob, device, "MobileNetV3")
    del mob; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("  Extracting Noise CNN features (128-dim)...")
    noise_cnn = NoiseStreamCNN(out_dim=128).to(device)
    noise_cnn.load_state_dict(
        torch.load(MODEL_B_NOISE, map_location=device, weights_only=True))
    noise_feats = extract_noise(frame_paths, noise_cnn, device)
    del noise_cnn; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("  Computing Farneback optical flow (5-dim)...")
    flow = compute_flow(frame_paths)

    X = np.hstack([mob_feats, noise_feats, flow])
    print(f"  Feature matrix: {X.shape}")
    X_sc     = scaler.transform(X)
    y_pred   = clf.predict(X_sc)
    y_proba  = clf.predict_proba(X_sc)[:, 1]
    return y_pred, y_proba


# ── OUTPUT ────────────────────────────────────────────────────────────────────

def print_results(video_name, branch, frame_paths, y_pred, y_proba):
    out = sys.stdout
    frame_names = [os.path.basename(p) for p in frame_paths]
    n_ai   = int((y_pred == 1).sum())
    n_real = int((y_pred == 0).sum())

    # Video-level: majority vote
    video_pred  = int(np.bincount(y_pred).argmax())
    video_label = "AI" if video_pred == 1 else "Real"
    video_conf  = float(np.mean(y_proba))

    # ── Per-frame table ───────────────────────────────────────────────────────
    out.write("\n")
    out.write("=" * 62 + "\n")
    out.write(f"  PER-FRAME PREDICTIONS  |  {video_name}  [{branch}]\n")
    out.write("=" * 62 + "\n")
    out.write(f"  {'Frame':<18}  {'Class':>5}  {'Verdict':>7}  {'P(AI)':>8}\n")
    out.write(f"  {'-'*18}  {'-'*5}  {'-'*7}  {'-'*8}\n")
    for name, pred, prob in zip(frame_names, y_pred, y_proba):
        verdict = "AI  " if pred == 1 else "Real"
        out.write(f"  {name:<18}  {pred:>5}  {verdict:>7}  {prob:>8.4f}\n")
    out.flush()

    # ── Video summary ─────────────────────────────────────────────────────────
    out.write("\n")
    out.write("=" * 62 + "\n")
    out.write(f"  FINAL VIDEO PREDICTION  |  {video_name}\n")
    out.write("=" * 62 + "\n")
    out.write(f"  Branch        : {branch}\n")
    out.write(f"  Total frames  : {len(y_pred)}\n")
    out.write(f"  Frames -> AI  : {n_ai}\n")
    out.write(f"  Frames -> Real: {n_real}\n")
    out.write(f"  Mean P(AI)    : {video_conf:.4f}\n")
    out.write(f"  VERDICT       : *** {video_label} ***  (class={video_pred})\n")
    out.write("=" * 62 + "\n")
    out.flush()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python demoinference.py <path/to/video.mp4 | video_name>")
        sys.exit(1)

    arg        = sys.argv[1]
    video_name = os.path.splitext(os.path.basename(arg))[0]
    start      = time.time()

    print(f"\n{'='*60}")
    print(f"  DEMO INFERENCE  —  {video_name}")
    print(f"{'='*60}")

    # 1. Look up routing CSV
    if not os.path.exists(ROUTING_CSV):
        print(f"Error: {ROUTING_CSV} not found. Run demorouter.py first.")
        sys.exit(1)
    routing = pd.read_csv(ROUTING_CSV)
    row     = routing[routing['video_name'] == video_name]
    if row.empty:
        print(f"Error: '{video_name}' not found in {ROUTING_CSV}.")
        print("Run:  python demorouter.py <path/to/video.mp4>  first.")
        sys.exit(1)
    branch     = row.iloc[0]['assigned_branch']
    pred_prob  = row.iloc[0]['pred_prob']
    print(f"\nRouting : {branch}  (router conf={pred_prob:.4f})")

    # 2. Collect frames
    frame_paths = collect_frames(video_name)
    if not frame_paths:
        print(f"Error: no frames found in {FRAMES_ROOT}/{video_name}/")
        print("Run:  python demorouter.py <path/to/video.mp4>  first.")
        sys.exit(1)
    print(f"Frames  : {len(frame_paths)}  from  {FRAMES_ROOT}/{video_name}/\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}\n")

    # 3. Branch-specific inference
    if branch == "Branch A":
        if not os.path.exists(MODEL_A_PKL):
            print(f"Error: {MODEL_A_PKL} not found.")
            sys.exit(1)
        y_pred, y_proba = infer_branch_a(frame_paths, device)
    else:
        y_pred, y_proba = infer_branch_b(frame_paths, device)

    # 4. Report
    print_results(video_name, branch, frame_paths, y_pred, y_proba)
    print(f"\nCompleted in {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
