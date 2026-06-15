import pandas as pd
import numpy as np
import os
import time
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix

# CONFIGURATION
INPUT_METADATA = "video_quality.csv"
OUTPUT_ROUTING = "video_routing.csv"
ROUTER_MODEL = "router_model.pkl"

# BALANCED HYPERPARAMETERS
ITERATIONS = 200           
LEARNING_RATE = 0.05       
MAX_DEPTH = 4              # Slightly deeper to capture "High Res" clusters
MIN_SAMPLES_LEAF = 4       # The "Goldilocks" value. 
                           # 1 = Memorization (Trap Fails). 
                           # 10 = Underfitting (40 Mismatches).
                           # 4 = Generalization (Trap Caught, Real Rules Kept).
STRICTNESS_PENALTY = 1.0

def route_videos():
    start=time.time()
    if not os.path.exists(INPUT_METADATA):
        print("Error: Quality CSV not found.")
        return

    df = pd.read_csv(INPUT_METADATA)
    
    # 1. GROUND TRUTH
    def get_target(subfolder):
        sub = str(subfolder).lower()
        if "high_quality" in sub: return 1
        if "low_quality" in sub: return 0
        return -1 

    df['target'] = df['subfolder'].apply(get_target)
    train_df = df[df['target'] != -1].copy()
    
    # 2. FEATURE ENGINEERING
    train_df['log_bitrate'] = np.log1p(train_df['bitrate'])
    train_df['log_bpp'] = np.log1p(train_df['bpp'])
    train_df['info_density'] = train_df['height'] * train_df['log_bitrate']
    
    features = ['height', 'bitrate', 'bpp', 'log_bitrate', 'log_bpp', 'info_density']
    X = train_df[features].values
    y = train_df['target'].values
    
    sample_weights = np.ones(len(y))
    sample_weights[y == 0] = STRICTNESS_PENALTY

    # 3. 5-FOLD CROSS-VALIDATION
    print(f"\n--- RUNNING 5-FOLD CROSS-VALIDATION ---")
    print(f"Constraints: Min Samples Leaf = {MIN_SAMPLES_LEAF}")
    
    clf = GradientBoostingClassifier(
        n_estimators=ITERATIONS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF, 
        subsample=0.8,
        random_state=42
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    
    print(f"Tests: {[f'{s:.1%}' for s in scores]}")
    avg_acc = scores.mean()
    print(f"EXPECTED ACCURACY: {avg_acc:.1%}")

    # 4. FINAL TRAINING
    print("\n--- Training Final Production Model ---")
    clf.fit(X, y, sample_weight=sample_weights)
    
    # 5. PREDICT
    df['log_bitrate'] = np.log1p(df['bitrate'])
    df['log_bpp'] = np.log1p(df['bpp'])
    df['info_density'] = df['height'] * df['log_bitrate']
    X_full = df[features].values
    
    df['pred_class'] = clf.predict(X_full)
    df['pred_prob'] = clf.predict_proba(X_full)[:, 1]
    
    df['assigned_branch'] = df['pred_class'].apply(
        lambda x: "Branch A" if x == 1 else "Branch B"
    )
    
    # 6. REPORT
    def check_status(row):
        target = row['target']
        if target == -1: return "Unknown"
        return "Correct" if target == row['pred_class'] else "Mismatch"

    df['validation_status'] = df.apply(check_status, axis=1)
    valid = df[df['validation_status'] != "Unknown"]
    
    print("\nFinal Confusion Matrix:")
    cm = confusion_matrix(valid['target'], valid['pred_class'])
    print(f"True LQ -> Routed LQ:  {cm[0][0]}")
    print(f"True LQ -> Routed HQ:  {cm[0][1]}")
    print(f"True HQ -> Routed LQ:  {cm[1][0]}")
    print(f"True HQ -> Routed HQ:  {cm[1][1]}")
    
    mismatches = valid[valid['validation_status'] == "Mismatch"]
    if len(mismatches) > 0:
        print(f"\nWARNING: Found {len(mismatches)} routing mismatches (low confidence or ambiguous quality).")
        cols = ['video_name', 'subfolder', 'assigned_branch', 'height', 'bitrate', 'pred_prob']
        print(mismatches[cols].sort_values(by='pred_prob').to_string(index=False))
    else:
        print("\nZero mismatches — all videos routed consistently with their folder labels.")

    # 7. SAVE
    df.to_csv(OUTPUT_ROUTING, index=False)
    joblib.dump({'model': clf, 'features': features}, ROUTER_MODEL)
    print(f"\nRouter model saved to: {ROUTER_MODEL}")
    end=time.time()
    print(f"Routing completed in {end-start:.2f} seconds.")

if __name__ == "__main__":
    route_videos()