import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

# =============================================================================
# FOLD-AWARE DUAL-STREAM EXTRACTOR (Branch A) — STEP 1 of 3
# -----------------------------------------------------------------------------
# This extractor runs ONLY the frozen EfficientNet-B0 backbone and caches the
# 1280-dim semantic features per frame. The noise CNN is NOT trained here.
#
# WHY: in the original dualstream extractor, the noise CNN was fine-tuned on
# 80% of the videos before features were saved. When 5-fold CV later split
# those features differently, test folds contained videos the noise CNN had
# already seen during fine-tuning. This is feature-extractor leakage.
#
# In this fold-aware pipeline, the noise CNN is trained INSIDE the trainer,
# fresh per fold, on that fold's training videos only. The frozen EfficientNet
# never trains, so its features are leakage-free regardless of how the data
# is later split — that's why it can stay in this extractor (computed once).
# =============================================================================

INPUT_ROOT      = "processed"
ROUTING_FILE    = "video_routing.csv"
OUTPUT_FEATURES = "branch_a_features_dualstream_fa.npy"   # 1280-dim semantic only
OUTPUT_METADATA = "branch_a_metadata_dualstream_fa.csv"
BATCH_SIZE      = 32


def get_extractor():
    print("Loading frozen EfficientNet-B0 (semantic stream only)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    class EffExtractor(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.features = m.features
            self.avgpool  = m.avgpool
            self.flatten  = nn.Flatten()
        def forward(self, x):
            return self.flatten(self.avgpool(self.features(x)))

    ex = EffExtractor(model).to(device).eval()
    preprocess = weights.transforms()
    return ex, preprocess, device


def main():
    start = time.time()

    if not os.path.exists(ROUTING_FILE):
        print(f"Error: {ROUTING_FILE} not found. Run router.py first.")
        return

    df = pd.read_csv(ROUTING_FILE)
    branch_a_df = df[df['assigned_branch'] == "Branch A"].copy()
    valid = set(branch_a_df['video_name'].unique())

    print(f"Total Videos in Router: {len(df)}")
    print(f"Branch A (HQ) Videos: {len(valid)}")
    if len(valid) == 0:
        print("No Branch A videos found.")
        return

    # Scan processed/ for Branch A frames
    print(f"Scanning '{INPUT_ROOT}' for Branch A frames...")
    paths = []
    matched = set()
    for root, _, files in os.walk(INPUT_ROOT):
        folder = os.path.basename(root)
        if folder in valid:
            matched.add(folder)
            row = branch_a_df[branch_a_df['video_name'] == folder].iloc[0]
            vid_meta = {
                'video_name':      folder,
                'label':           row['label'],
                'assigned_branch': 'Branch A',
                'subfolder':       row['subfolder'],
            }
            for f in sorted(files):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    entry = vid_meta.copy()
                    entry['filename']   = f
                    entry['frame_path'] = os.path.join(root, f)
                    entry['key']        = f"{folder}__{f}"
                    paths.append(entry)

    missing = valid - matched
    if missing:
        print(f"WARNING: {len(missing)} Branch A videos missing: {sorted(missing)}")
    else:
        print(f"All {len(valid)} Branch A videos located.")
    print(f"Found {len(paths)} frames.")
    if not paths:
        return

    ex, preprocess, device = get_extractor()

    # Extract semantic features, keeping metadata in sync with successful loads
    print("Extracting frozen EfficientNet semantic features (1280-dim)...")
    n_batches = (len(paths) + BATCH_SIZE - 1) // BATCH_SIZE
    features_list = []
    metadata_list = []

    skipped = []
    for i in tqdm(range(0, len(paths), BATCH_SIZE), total=n_batches):
        batch_meta = paths[i:i+BATCH_SIZE]
        batch_tensors = []
        ok_meta = []
        for m in batch_meta:
            try:
                img = Image.open(m['frame_path']).convert('RGB')
                batch_tensors.append(preprocess(img))
                ok_meta.append(m)
            except Exception as e:
                skipped.append((m['frame_path'], str(e)))
        if not batch_tensors:
            continue
        batch = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            out = ex(batch).cpu().numpy()
        features_list.append(out)
        metadata_list.extend(ok_meta)

    # LOUD skip report: small numbers are noise; large numbers mean something is wrong.
    if skipped:
        rate = len(skipped) / len(paths)
        print(f"\n[SKIP REPORT] {len(skipped)}/{len(paths)} frames failed to load ({rate:.1%}).")
        for p, e in skipped[:20]:
            print(f"   - {p}: {e}")
        if len(skipped) > 20:
            print(f"   ... and {len(skipped)-20} more.")
        # Refuse to silently proceed on a large drop — likely a path/dataset bug.
        if rate > 0.02:
            print(f"\nERROR: skip rate {rate:.1%} exceeds 2% threshold. Aborting so this")
            print("       is not silently absorbed. Fix the frame paths, then re-run.")
            return
    else:
        print("\n[SKIP REPORT] 0 frames skipped — all frames loaded.")

    if not features_list:
        print("Extraction failed.")
        return

    final_features = np.vstack(features_list)
    meta_df = pd.DataFrame(metadata_list)

    assert len(final_features) == len(meta_df), \
        f"row-count mismatch after extraction: {len(final_features)} vs {len(meta_df)}"

    np.save(OUTPUT_FEATURES, final_features)
    meta_df.to_csv(OUTPUT_METADATA, index=False)

    print(f"\nSaved semantic features: {OUTPUT_FEATURES}  shape={final_features.shape}")
    print(f"Saved metadata:           {OUTPUT_METADATA}")
    print(f"\n--- Sanity Report ---")
    print(f"Frames: {len(meta_df)}  Videos: {meta_df['video_name'].nunique()}")
    print(meta_df['label'].value_counts().to_string())
    print(f"\nNote: this file contains ONLY the 1280-dim semantic stream.")
    print(f"The noise CNN is trained per-fold inside the trainer.")
    print(f"Total time: {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
