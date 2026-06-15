import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import time


# MATCHED-SIZE NO-BRANCHING BASELINE — EXTRACTOR (STEP 1 of 3, BRANCH B TRIPLE)

# Selects 65 videos at random from each of the 4 subfolders
# (high_quality_ai, high_quality_real, low_quality_ai, low_quality_real)
# = 260 videos total, matching the per-branch training size of Branch B.
#
# Extracts MobileNetV3-Large (960-dim) semantic features only.
# The Noise CNN (128-dim) is fold-aware and trained fresh per fold at training
# time. Farneback optical flow (5-dim) is also computed at training time.
# Final feature vector at training: 960 + 128 + 5 = 1093-dim.
#
# Outputs:
#   mixed260_features_b_triplestream.npy   — [N, 960] MobileNetV3 features
#   mixed260_metadata_b_triplestream.csv   — shared metadata (frame_path, label, etc.)
# =============================================================================

INPUT_ROOT   = "processed"
ROUTING_FILE = "video_routing.csv"

OUTPUT_FEATURES = "mixed260_features_b_triplestream.npy"
OUTPUT_METADATA = "mixed260_metadata_b_triplestream.csv"

BATCH_SIZE  = 32
PER_BUCKET  = 65
SAMPLE_SEED = 44

SUBFOLDERS = ['high_quality_ai', 'high_quality_real',
              'low_quality_ai',  'low_quality_real']


def get_feature_extractor(device):
    weights  = models.MobileNet_V3_Large_Weights.DEFAULT
    base     = models.mobilenet_v3_large(weights=weights)

    class MobileNetExtractor(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.features = m.features
            self.avgpool  = m.avgpool
            self.flatten  = nn.Flatten()
        def forward(self, x):
            return self.flatten(self.avgpool(self.features(x)))

    extractor  = MobileNetExtractor(base).to(device).eval()
    preprocess = weights.transforms()
    return extractor, preprocess


def main():
    start = time.time()

    if not os.path.exists(ROUTING_FILE):
        print(f"Error: {ROUTING_FILE} not found. Run router.py first.")
        return

    df = pd.read_csv(ROUTING_FILE)

    # 1. Balanced subsample: 65 per subfolder, fixed seed
    rng = np.random.RandomState(SAMPLE_SEED)
    selected_videos = []

    print(f"Sampling {PER_BUCKET} videos from each of {len(SUBFOLDERS)} subfolders "
          f"(seed={SAMPLE_SEED})...")
    for sf in SUBFOLDERS:
        bucket = df[df['subfolder'] == sf]['video_name'].unique()
        if len(bucket) < PER_BUCKET:
            print(f"ERROR: subfolder '{sf}' has only {len(bucket)} videos, "
                  f"need {PER_BUCKET}. Aborting.")
            return
        chosen = rng.choice(bucket, size=PER_BUCKET, replace=False)
        selected_videos.extend(chosen)
        print(f"   {sf}: {len(bucket)} available -> selected {PER_BUCKET}")

    selected_set = set(selected_videos)
    print(f"\nTotal selected: {len(selected_set)} videos "
          f"(should be {PER_BUCKET * len(SUBFOLDERS)})")

    # 2. Collect frame paths for selected videos
    print(f"\nScanning '{INPUT_ROOT}' for frames belonging to selected videos...")
    paths = []
    matched_videos = set()

    for root, dirs, files in os.walk(INPUT_ROOT):
        folder_name = os.path.basename(root)
        if folder_name in selected_set:
            matched_videos.add(folder_name)
            vid_meta = df[df['video_name'] == folder_name].iloc[0].to_dict()
            for file in sorted(files):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, file)
                    meta_entry = vid_meta.copy()
                    meta_entry['filename']   = file
                    meta_entry['frame_path'] = full_path
                    paths.append((full_path, meta_entry))

    missing = selected_set - matched_videos
    if missing:
        print(f"WARNING: {len(missing)} selected videos have no frames in '{INPUT_ROOT}':")
        print(f"   {sorted(missing)}")
    else:
        print(f"All {len(selected_set)} selected videos located.")
    print(f"Found {len(paths)} frames.")

    if len(paths) == 0:
        print("Error: No frames found.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model, preprocess = get_feature_extractor(device)
    print("Loaded MobileNetV3-Large (960-dim).")

    features_list = []
    metadata_list = []

    num_batches = (len(paths) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\nExtracting MobileNetV3 features for {len(paths)} frames...")

    for i in tqdm(range(0, len(paths), BATCH_SIZE), total=num_batches):
        batch_data    = paths[i : i + BATCH_SIZE]
        batch_tensors = []

        for fpath, meta in batch_data:
            try:
                img = Image.open(fpath).convert('RGB')
                batch_tensors.append(preprocess(img))
                metadata_list.append(meta)
            except Exception as e:
                print(f"Error loading {fpath}: {e}")

        if not batch_tensors:
            continue

        with torch.no_grad():
            out = model(torch.stack(batch_tensors).to(device))

        features_list.append(out.cpu().numpy())

    if not features_list:
        print("Extraction failed — no features produced.")
        return

    final_features = np.vstack(features_list)
    meta_df        = pd.DataFrame(metadata_list)

    print(f"\nFeature matrix shape: {final_features.shape}")
    print(f"Metadata rows:        {len(meta_df)}")

    np.save(OUTPUT_FEATURES, final_features)
    meta_df.to_csv(OUTPUT_METADATA, index=False)

    print(f"\nSaved {OUTPUT_FEATURES}")
    print(f"Saved {OUTPUT_METADATA}")

    print(f"\n--- Sanity Report ---")
    print(f"Frames extracted:  {len(meta_df)}")
    print(f"Unique videos:     {meta_df['video_name'].nunique()}")
    print(f"Label distribution:\n{meta_df['label'].value_counts().to_string()}")
    print(f"Subfolder distribution (should be balanced):\n"
          f"{meta_df['subfolder'].value_counts().to_string()}")

    print(f"\nTotal time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
