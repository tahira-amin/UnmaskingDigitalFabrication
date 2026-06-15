import pandas as pd
import numpy as np
import os
import time

# =============================================================================
# FOLD-AWARE DUAL-STREAM LABELER (Branch A) — STEP 2 of 3
# -----------------------------------------------------------------------------
# Adds the target column to the semantic-feature metadata. The features file
# itself (1280-dim semantic) is unchanged. frame_path is carried through so the
# trainer can load images to train the noise CNN per fold.
# =============================================================================

INPUT_FEATURES          = "branch_a_features_dualstream_fa_mobilenet.npy"
INPUT_METADATA          = "branch_a_metadata_dualstream_fa_mobilenet.csv"
OUTPUT_METADATA_LABELED = "branch_a_metadata_labeled_dualstream_fa_mobilenet.csv"


def main():
    start = time.time()

    if not os.path.exists(INPUT_FEATURES) or not os.path.exists(INPUT_METADATA):
        print("Error: Missing features or metadata.")
        print(f"   Expected: {INPUT_FEATURES}")
        print(f"   Expected: {INPUT_METADATA}")
        print("Run feature_extractor_a_dualstream_fa.py first.")
        return

    print("Loading semantic features + metadata for Branch A (fold-aware)...")
    features = np.load(INPUT_FEATURES)   # [N, 1280] semantic only
    metadata = pd.read_csv(INPUT_METADATA)
    print(f"Loaded {len(features)} frames with {features.shape[1]}-dim semantic features.")

    if len(features) != len(metadata):
        print(f"ERROR: Feature count ({len(features)}) != Metadata count ({len(metadata)})")
        return

    if 'frame_path' not in metadata.columns:
        print("ERROR: metadata is missing 'frame_path' (needed for per-fold noise CNN training).")
        return
    if 'key' not in metadata.columns:
        print("ERROR: metadata is missing the 'key' column. This usually means the CSV is")
        print("       from an OLD extractor run, not feature_extractor_a_dualstream_fa.py.")
        print("       Re-run the fold-aware extractor so features and metadata are from the")
        print("       SAME run, then re-run this labeler.")
        return

    metadata['target'] = metadata['label'].apply(lambda x: 1 if str(x).lower() == 'ai' else 0)
    metadata['branch'] = 'Branch A'
    metadata.to_csv(OUTPUT_METADATA_LABELED, index=False)

    print(f"\n[DONE] Prepared {len(features)} frames for fold-aware training")
    print(f"   Semantic features: {INPUT_FEATURES} (1280-dim, noise CNN trained per fold)")
    print(f"   Labeled metadata:  {OUTPUT_METADATA_LABELED}")
    print(f"   Total time: {time.time()-start:.2f}s")

    print("\n--- Class Distribution ---")
    print(metadata['label'].value_counts())

    ai_mismatch   = ((metadata['label'].str.lower() == 'ai')   & (metadata['target'] != 1)).sum()
    real_mismatch = ((metadata['label'].str.lower() == 'real') & (metadata['target'] != 0)).sum()
    if ai_mismatch + real_mismatch > 0:
        print(f"\nWARNING: target/label mismatch — AI: {ai_mismatch}, Real: {real_mismatch}")
    else:
        print("\nTarget/label encoding verified — no mismatches.")


if __name__ == "__main__":
    main()
