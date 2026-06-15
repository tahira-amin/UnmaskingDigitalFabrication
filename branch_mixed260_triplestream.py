import pandas as pd
import numpy as np
import os
import time

# =============================================================================
# MATCHED-SIZE NO-BRANCHING BASELINE — LABELER (STEP 2 of 3, TRIPLE-STREAM)
# -----------------------------------------------------------------------------
# Reads the shared metadata produced by feature_extractor_mixed260_triplestream.py,
# adds a target column derived from the label, and marks branch as 'Mixed260Triple'.
# Feature .npy files are unchanged — only the metadata CSV is updated.
# =============================================================================

INPUT_EFFNET    = "mixed260_features_effnet_triple.npy"
INPUT_MOBILENET = "mixed260_features_mobilenet_triple.npy"
INPUT_METADATA  = "mixed260_metadata_triple.csv"

OUTPUT_METADATA_LABELED = "mixed260_metadata_labeled_triple.csv"


def main():
    start = time.time()

    for f in [INPUT_EFFNET, INPUT_MOBILENET, INPUT_METADATA]:
        if not os.path.exists(f):
            print(f"Error: {f} not found.")
            print("Run feature_extractor_mixed260_triplestream.py first.")
            return

    print("Loading mixed-260 triple-stream features and metadata...")
    eff    = np.load(INPUT_EFFNET)
    mobile = np.load(INPUT_MOBILENET)
    meta   = pd.read_csv(INPUT_METADATA)

    print(f"EfficientNet features:  {eff.shape}")
    print(f"MobileNet features:     {mobile.shape}")
    print(f"Metadata rows:          {len(meta)}")

    if not (len(eff) == len(mobile) == len(meta)):
        print(f"ERROR: Row count mismatch — EffNet:{len(eff)}, "
              f"MobileNet:{len(mobile)}, Metadata:{len(meta)}")
        return

    meta['target'] = meta['label'].apply(
        lambda x: 1 if str(x).strip().lower() == 'ai' else 0
    )
    meta['branch'] = 'Mixed260Triple'

    meta.to_csv(OUTPUT_METADATA_LABELED, index=False)
    print(f"\nSaved labeled metadata to {OUTPUT_METADATA_LABELED}")

    print(f"\n--- Class Distribution ---")
    print(meta['label'].value_counts().to_string())

    print(f"\n--- Subfolder Distribution (should be 4 x equal) ---")
    print(meta['subfolder'].value_counts().to_string())

    ai_mismatch   = ((meta['label'].str.lower() == 'ai')   & (meta['target'] != 1)).sum()
    real_mismatch = ((meta['label'].str.lower() == 'real') & (meta['target'] != 0)).sum()
    if ai_mismatch + real_mismatch > 0:
        print(f"\nWARNING: target/label mismatch — AI: {ai_mismatch}, Real: {real_mismatch}")
    else:
        print(f"\nTarget/label encoding verified — no mismatches.")

    print(f"\nTotal time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
