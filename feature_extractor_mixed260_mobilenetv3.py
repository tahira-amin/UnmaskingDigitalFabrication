import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

# =============================================================================
# MATCHED-SIZE NO-BRANCHING BASELINE — EXTRACTOR (STEP 1 of 3)
# -----------------------------------------------------------------------------
# Selects 65 videos at random from each of the 4 subfolders
# (high_quality_ai, high_quality_real, low_quality_ai, low_quality_real)
# = 260 videos total, matching the per-branch training size of Branch A and
# Branch B. This isolates the branching effect from the data-size effect that
# made the all-520 baseline look better than the branches.
# Mirrors feature_extractor_all_effnet.py except for the balanced subsample.
# =============================================================================

# CONFIGURATION
INPUT_ROOT = "processed"
ROUTING_FILE = "video_routing.csv"
OUTPUT_FEATURES = "mixed260_features_mobilenetv3.npy"
OUTPUT_METADATA = "mixed260_metadata_mobilenetv3.csv"
BATCH_SIZE = 32

# Balanced subsample: 65 videos per subfolder x 4 subfolders = 260 total
PER_BUCKET = 65
SAMPLE_SEED = 42   # change and re-run if you want a sanity check on a different draw

SUBFOLDERS = ['high_quality_ai', 'high_quality_real',
              'low_quality_ai',  'low_quality_real']


def get_feature_extractor():
    print("Loading EfficientNet-B0 (Pre-trained)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    weights = models.MobileNet_V3_Large_Weights.DEFAULT
    model = models.mobilenet_v3_large(weights=weights)

    class EfficientNetExtractor(nn.Module):
        def __init__(self, original_model):
            super(EfficientNetExtractor, self).__init__()
            self.features = original_model.features
            self.avgpool = original_model.avgpool
            self.flatten = nn.Flatten()

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = self.flatten(x)
            return x

    extractor = EfficientNetExtractor(model)
    extractor.to(device)
    extractor.eval()

    preprocess = weights.transforms()
    return extractor, preprocess, device


def main():
    start_time = time.time()

    if not os.path.exists(ROUTING_FILE):
        print(f"Error: {ROUTING_FILE} not found. Run router.py first.")
        return

    df = pd.read_csv(ROUTING_FILE)

    # 1. Balanced subsample: 65 per subfolder, fixed seed for reproducibility
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

    # 2. Find Frames for the selected videos only
    print(f"\nScanning '{INPUT_ROOT}' for frames belonging to selected videos...")
    paths = []
    matched_videos = set()

    for root, dirs, files in os.walk(INPUT_ROOT):
        folder_name = os.path.basename(root)
        if folder_name in selected_set:
            matched_videos.add(folder_name)
            vid_meta = df[df['video_name'] == folder_name].iloc[0].to_dict()
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, file)
                    meta_entry = vid_meta.copy()
                    meta_entry['filename'] = file
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

    # 3. Setup EfficientNet
    model, preprocess, device = get_feature_extractor()

    features_list = []
    metadata_list = []

    # 4. Extraction Loop
    print("Extracting EfficientNet-B0 features (1280 dimensions)...")
    num_batches = (len(paths) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in tqdm(range(0, len(paths), BATCH_SIZE), total=num_batches):
        batch_data = paths[i : i + BATCH_SIZE]
        batch_tensors = []

        for fpath, meta in batch_data:
            try:
                img = Image.open(fpath).convert('RGB')
                input_tensor = preprocess(img)
                batch_tensors.append(input_tensor)
                metadata_list.append(meta)
            except Exception as e:
                print(f"Error loading {fpath}: {e}")

        if not batch_tensors:
            continue

        batch_stack = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            outputs = model(batch_stack)

        features_list.append(outputs.cpu().numpy())

    # 5. Save Results
    if features_list:
        final_features = np.vstack(features_list)
        print(f"Feature Matrix Shape: {final_features.shape}")

        np.save(OUTPUT_FEATURES, final_features)

        meta_df = pd.DataFrame(metadata_list)
        meta_df.to_csv(OUTPUT_METADATA, index=False)

        print(f"Saved features to {OUTPUT_FEATURES}")
        print(f"Saved metadata to {OUTPUT_METADATA}")

        # Sanity report
        print(f"\n--- Sanity Report ---")
        print(f"Frames extracted: {len(meta_df)}")
        print(f"Unique videos: {meta_df['video_name'].nunique()}")
        print(f"Label distribution:")
        print(meta_df['label'].value_counts().to_string())
        print(f"Subfolder distribution (should be balanced):")
        print(meta_df['subfolder'].value_counts().to_string())
        if 'assigned_branch' in meta_df.columns:
            print(f"Branch distribution (from router — reference only, not used):")
            print(meta_df['assigned_branch'].value_counts().to_string())
    else:
        print("Extraction failed.")

    print(f"\nTotal Execution Time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
