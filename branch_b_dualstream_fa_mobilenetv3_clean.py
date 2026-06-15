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
# FOLD-AWARE DUAL-STREAM EXTRACTOR (Branch B) — STEP 1 of 2
# -----------------------------------------------------------------------------
# This extractor runs ONLY the frozen MobileNetV3-Large backbone and caches the
# 960-dim semantic features per frame. The noise CNN is NOT trained here.
#
# WHY: in the original dualstream extractor, the noise CNN was fine-tuned before
# features were saved. When 5-fold CV later split those saved features differently,
# test folds could contain videos the noise CNN had already seen during fine-tuning.
# This is feature-extractor leakage.
#
# In this fold-aware pipeline, the noise CNN is trained INSIDE the trainer,
# fresh per fold, on that fold's training videos only. The frozen MobileNetV3
# never trains, so its features are leakage-free regardless of how the data
# is later split — that's why it can stay in this extractor, computed once.
#
# Branch B semantic stream:
#   MobileNetV3-Large -> 960-dim features
# =============================================================================

INPUT_ROOT      = "processed"
ROUTING_FILE    = "video_routing.csv"
OUTPUT_FEATURES = "branch_b_features_dualstream_fa.npy"   # 960-dim MobileNetV3 semantic only
OUTPUT_METADATA = "branch_b_metadata_dualstream_fa.csv"
BATCH_SIZE      = 32


def normalize_name(name: str) -> str:
    """Normalize video/folder names for safer matching against processed folders."""
    base = os.path.splitext(str(name))[0]
    return base.lower().strip()


def get_extractor():
    print("Loading frozen MobileNetV3-Large (semantic stream only)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # MobileNetV3-Large ImageNet weights
    weights = models.MobileNet_V3_Large_Weights.DEFAULT
    model = models.mobilenet_v3_large(weights=weights)

    class MobileNetExtractor(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.features = m.features
            self.avgpool  = m.avgpool
            self.flatten  = nn.Flatten()

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = self.flatten(x)
            return x  # [B, 960]

    extractor = MobileNetExtractor(model).to(device).eval()
    preprocess = weights.transforms()

    # Return MobileNetV3 weights object as well, so the saved pipeline is explicit.
    return extractor, preprocess, device, weights


def main():
    start = time.time()

    if not os.path.exists(ROUTING_FILE):
        print(f"Error: {ROUTING_FILE} not found. Run router.py first.")
        return

    df = pd.read_csv(ROUTING_FILE)

    required_cols = {'video_name', 'assigned_branch', 'label'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        print(f"Error: {ROUTING_FILE} is missing required columns: {sorted(missing_cols)}")
        return

    branch_b_df = df[df['assigned_branch'] == "Branch B"].copy()

    if len(branch_b_df) == 0:
        print("No Branch B videos found in router.")
        return

    # Build a normalized lookup so processed folder names can match video names
    # even if extensions/capitalization differ. Keep original video_name in metadata.
    target_videos = {}
    duplicate_keys = []

    for _, row in branch_b_df.iterrows():
        norm_name = normalize_name(row['video_name'])
        if norm_name in target_videos:
            duplicate_keys.append(norm_name)

        meta = {
            'video_name':      row['video_name'],
            'label':           row['label'],
            'assigned_branch': 'Branch B',
        }

        if 'subfolder' in row.index:
            meta['subfolder'] = row['subfolder']

        target_videos[norm_name] = meta

    if duplicate_keys:
        print(f"WARNING: duplicate normalized Branch B video names detected: {sorted(set(duplicate_keys))[:10]}")
        print("         One routing row may overwrite another. Check video_routing.csv if this looks wrong.")

    print(f"Total Videos in Router: {len(df)}")
    print(f"Branch B (LQ) Videos: {len(target_videos)}")

    # Scan processed/ for Branch B frames
    print(f"Scanning '{INPUT_ROOT}' for Branch B frames...")
    paths = []
    matched = set()

    for root, _, files in os.walk(INPUT_ROOT):
        folder = os.path.basename(root)
        norm_folder = normalize_name(folder)

        if norm_folder in target_videos:
            matched.add(norm_folder)
            vid_meta = target_videos[norm_folder].copy()

            for f in sorted(files):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    entry = vid_meta.copy()
                    entry['filename']   = f
                    entry['frame_path'] = os.path.join(root, f)
                    entry['key']        = f"{vid_meta['video_name']}__{f}"
                    paths.append(entry)

    print(f"Found {len(paths)} Branch B frames from {len(matched)}/{len(target_videos)} videos.")

    missing = sorted(set(target_videos.keys()) - matched)
    if missing:
        print(f"WARNING: {len(missing)} Branch B videos were not found in processed/.")
        print(f"First missing examples: {missing[:10]}")

    if len(paths) == 0:
        print("Error: No Branch B frames found.")
        return

    extractor, preprocess, device, weights = get_extractor()
    print(f"Using weights: {weights}")

    features_list = []
    metadata_list = []

    print("Extracting frozen MobileNetV3 semantic features (960-dim)...")
    num_batches = (len(paths) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in tqdm(range(0, len(paths), BATCH_SIZE), total=num_batches):
        batch_meta = paths[i:i + BATCH_SIZE]
        batch_tensors = []
        valid_meta = []

        for meta in batch_meta:
            try:
                img = Image.open(meta['frame_path']).convert('RGB')
                batch_tensors.append(preprocess(img))
                valid_meta.append(meta)
            except Exception as e:
                print(f"Error loading {meta['frame_path']}: {e}")

        if not batch_tensors:
            continue

        batch_stack = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            outputs = extractor(batch_stack)

        features_list.append(outputs.cpu().numpy())
        metadata_list.extend(valid_meta)

    if not features_list:
        print("Extraction failed: no valid image batches.")
        return

    final_features = np.vstack(features_list)
    meta_df = pd.DataFrame(metadata_list)

    if len(final_features) != len(meta_df):
        print(f"ERROR: Feature/metadata mismatch before save ({len(final_features)} vs {len(meta_df)})")
        return

    if 'key' not in meta_df.columns:
        print("ERROR: key column missing before save.")
        return

    if meta_df['key'].duplicated().any():
        dups = meta_df['key'][meta_df['key'].duplicated()].unique()[:5]
        print(f"ERROR: duplicate frame keys detected, examples: {list(dups)}")
        return

    np.save(OUTPUT_FEATURES, final_features)
    meta_df.to_csv(OUTPUT_METADATA, index=False)

    print(f"\nSaved semantic features to {OUTPUT_FEATURES} with shape {final_features.shape}")
    print(f"Saved metadata to {OUTPUT_METADATA}")
    print(f"Total time: {time.time() - start:.2f}s")
    print("\nNote: this file contains ONLY the 960-dim MobileNetV3 semantic stream.")
    print("      The SRM/noise CNN is trained inside train_branch_b_dualstream_fa_no_temporal.py.")


if __name__ == "__main__":
    main()
