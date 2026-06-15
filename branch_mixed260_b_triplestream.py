import pandas as pd
import numpy as np
import os
import time

# =============================================================================
# MATCHED-SIZE NO-BRANCHING BASELINE — LABELER (STEP 2 of 3, BRANCH B TRIPLE)
# -----------------------------------------------------------------------------
# Reads the metadata produced by feature_extractor_mixed260_b_triplestream.py,
# adds a target column derived from the label, and marks branch as
# 'Mixed260BTriple'. The feature .npy file is unchanged.
# =============================================================================

INPUT_FEATURES          = "mixed260_features_b_triplestream.npy"
INPUT_METADATA          = "mixed260_metadata_b_triplestream.csv"
OUTPUT_METADATA_LABELED = "mixed260_metadata_labeled_b_triplestream.csv"


def main():
    start = time.time()

    for f in [INPUT_FEATURES, INPUT_METADATA]:
        if not os.path.exists(f):
            print(f"Error: {f} not found.")
            print("Run feature_extractor_mixed260_b_triplestream.py first.")
            return

    print("Loading mixed-260 Branch B triple-stream features and metadata...")
    features = np.load(INPUT_FEATURES)
    metadata = pd.read_csv(INPUT_METADATA)

    print(f"MobileNetV3 features: {features.shape}")
    print(f"Metadata rows:        {len(metadata)}")

    if len(features) != len(metadata):
        print(f"ERROR: Feature count ({len(features)}) != Metadata count ({len(metadata)})")
        return

    metadata['target'] = metadata['label'].apply(
        lambda x: 1 if str(x).strip().lower() == 'ai' else 0
    )
    metadata['branch'] = 'Mixed260BTriple'

    metadata.to_csv(OUTPUT_METADATA_LABELED, index=False)
    print(f"\nSaved labeled metadata to {OUTPUT_METADATA_LABELED}")

    print(f"\n--- Class Distribution ---")
    print(metadata['label'].value_counts().to_string())

    print(f"\n--- Subfolder Distribution (should be 4 x equal) ---")
    print(metadata['subfolder'].value_counts().to_string())

    ai_mismatch   = ((metadata['label'].str.lower() == 'ai')   & (metadata['target'] != 1)).sum()
    real_mismatch = ((metadata['label'].str.lower() == 'real') & (metadata['target'] != 0)).sum()
    if ai_mismatch + real_mismatch > 0:
        print(f"\nWARNING: target/label mismatch — AI: {ai_mismatch}, Real: {real_mismatch}")
    else:
        print(f"\nTarget/label encoding verified — no mismatches.")

    print(f"\nTotal time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
