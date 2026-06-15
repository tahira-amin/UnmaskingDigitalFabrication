import pandas as pd
import numpy as np
import os
import time

# =============================================================================
# MATCHED-SIZE NO-BRANCHING BASELINE — LABELER (STEP 2 of 3)
# -----------------------------------------------------------------------------
# Mirrors branch_all_effnet.py. Adds target column from label, marks branch
# as 'Mixed260' (65 per subfolder x 4 = 260 videos, balanced HQ/LQ + AI/Real).
# =============================================================================

INPUT_FEATURES          = "mixed260_features_effnet.npy"
INPUT_METADATA          = "mixed260_metadata_effnet.csv"
OUTPUT_FEATURES         = "mixed260_features_effnet.npy"     # features unchanged
OUTPUT_METADATA_LABELED = "mixed260_metadata_labeled_effnet.csv"


def main():
    start_time = time.time()

    if not os.path.exists(INPUT_FEATURES) or not os.path.exists(INPUT_METADATA):
        print("Error: Missing features or metadata.")
        print(f"   Expected: {INPUT_FEATURES}")
        print(f"   Expected: {INPUT_METADATA}")
        print("Run feature_extractor_mixed260_effnet.py first.")
        return

    print("Loading EfficientNet-B0 features for mixed-260 baseline...")
    features = np.load(INPUT_FEATURES)
    metadata = pd.read_csv(INPUT_METADATA)

    print(f"Loaded {len(features)} frames with {features.shape[1]}-dimensional features.")

    if len(features) != len(metadata):
        print(f"ERROR: Feature count ({len(features)}) != Metadata count ({len(metadata)})")
        return

    metadata['target'] = metadata['label'].apply(lambda x: 1 if str(x).lower() == 'ai' else 0)
    metadata['branch'] = 'Mixed260'

    metadata.to_csv(OUTPUT_METADATA_LABELED, index=False)

    print(f"\n[DONE] Prepared {len(features)} frames for training")
    print(f"   Features: {OUTPUT_FEATURES} (1280-dim EfficientNet-B0)")
    print(f"   Metadata: {OUTPUT_METADATA_LABELED}")
    print(f"   Total Execution Time: {time.time() - start_time:.2f} seconds")

    print("\n--- Class Distribution ---")
    print(metadata['label'].value_counts())

    print("\n--- Subfolder Distribution (should be 4 x equal) ---")
    print(metadata['subfolder'].value_counts())

    # Verify target/label encoding
    ai_mismatch   = ((metadata['label'].str.lower() == 'ai')   & (metadata['target'] != 1)).sum()
    real_mismatch = ((metadata['label'].str.lower() == 'real') & (metadata['target'] != 0)).sum()
    if ai_mismatch + real_mismatch > 0:
        print(f"\nWARNING: target/label mismatch — AI: {ai_mismatch}, Real: {real_mismatch}")
    else:
        print(f"\nTarget/label encoding verified — no mismatches.")


if __name__ == "__main__":
    main()
