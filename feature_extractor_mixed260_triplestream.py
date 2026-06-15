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
# MATCHED-SIZE NO-BRANCHING BASELINE — EXTRACTOR (STEP 1 of 3, TRIPLE-STREAM)
# -----------------------------------------------------------------------------
# Selects 65 videos at random from each of the 4 subfolders
# (high_quality_ai, high_quality_real, low_quality_ai, low_quality_real)
# = 260 videos total, matching the per-branch training size of Branch A and B.
#
# Extracts BOTH EfficientNet-B0 (1280-dim) AND MobileNetV3-Large (960-dim)
# features in a single pass, guaranteeing perfect frame-order alignment.
# Optical flow is computed at training time (see train_mixed260_triplestream.py).
#
# Outputs:
#   mixed260_features_effnet_triple.npy     — [N, 1280]
#   mixed260_features_mobilenet_triple.npy  — [N, 960]
#   mixed260_metadata_triple.csv            — shared metadata (same row order)
# =============================================================================

INPUT_ROOT  = "processed"
ROUTING_FILE = "video_routing.csv"

OUTPUT_EFFNET    = "mixed260_features_effnet_triple.npy"
OUTPUT_MOBILENET = "mixed260_features_mobilenet_triple.npy"
OUTPUT_METADATA  = "mixed260_metadata_triple.csv"

BATCH_SIZE  = 32
PER_BUCKET  = 65
SAMPLE_SEED = 43

SUBFOLDERS = ['high_quality_ai', 'high_quality_real',
              'low_quality_ai',  'low_quality_real']


def load_effnet(device):
    weights = models.EfficientNet_B0_Weights.DEFAULT
    base    = models.efficientnet_b0(weights=weights)

    class EffNetExtractor(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.features = m.features
            self.avgpool  = m.avgpool
            self.flatten  = nn.Flatten()
        def forward(self, x):
            return self.flatten(self.avgpool(self.features(x)))

    extractor  = EffNetExtractor(base).to(device).eval()
    preprocess = weights.transforms()
    return extractor, preprocess


def load_mobilenet(device):
    weights = models.MobileNet_V3_Large_Weights.DEFAULT
    base    = models.mobilenet_v3_large(weights=weights)

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

    eff_model,    eff_pre    = load_effnet(device)
    mobile_model, mobile_pre = load_mobilenet(device)
    print("Loaded EfficientNet-B0 (1280-dim) and MobileNetV3-Large (960-dim).")

    eff_feats    = []
    mobile_feats = []
    metadata_list = []

    frame_paths = [p for p, _ in paths]
    metas       = [m for _, m in paths]

    num_batches = (len(frame_paths) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\nExtracting features for {len(frame_paths)} frames in {num_batches} batches...")

    for i in tqdm(range(0, len(frame_paths), BATCH_SIZE), total=num_batches):
        batch_paths = frame_paths[i : i + BATCH_SIZE]
        batch_metas = metas[i : i + BATCH_SIZE]

        eff_tensors    = []
        mobile_tensors = []
        valid_metas    = []

        for fpath, meta in zip(batch_paths, batch_metas):
            try:
                img = Image.open(fpath).convert('RGB')
                eff_tensors.append(eff_pre(img))
                mobile_tensors.append(mobile_pre(img))
                valid_metas.append(meta)
            except Exception as e:
                print(f"Error loading {fpath}: {e}")

        if not eff_tensors:
            continue

        with torch.no_grad():
            eff_out    = eff_model(torch.stack(eff_tensors).to(device)).cpu().numpy()
            mobile_out = mobile_model(torch.stack(mobile_tensors).to(device)).cpu().numpy()

        eff_feats.append(eff_out)
        mobile_feats.append(mobile_out)
        metadata_list.extend(valid_metas)

    if not eff_feats:
        print("Extraction failed — no features produced.")
        return

    final_eff    = np.vstack(eff_feats)
    final_mobile = np.vstack(mobile_feats)
    meta_df      = pd.DataFrame(metadata_list)

    print(f"\nEfficientNet features:  {final_eff.shape}")
    print(f"MobileNet features:     {final_mobile.shape}")
    print(f"Metadata rows:          {len(meta_df)}")

    np.save(OUTPUT_EFFNET,    final_eff)
    np.save(OUTPUT_MOBILENET, final_mobile)
    meta_df.to_csv(OUTPUT_METADATA, index=False)

    print(f"\nSaved {OUTPUT_EFFNET}")
    print(f"Saved {OUTPUT_MOBILENET}")
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
