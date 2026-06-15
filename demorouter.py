"""
demorouter.py  —  single-video pipeline
Usage:
    python demorouter.py <path/to/video.mp4>

Stages
  1. Extract frames  →  demo_frames/<video_name>/
  2. Assess quality  →  ffprobe metadata
  3. Route           →  router_model.pkl prediction
  4. Save            →  appends / updates demo_routing.csv
"""
import sys
import subprocess
import os
import json
import time
import numpy as np
import pandas as pd
import joblib

FRAMES_ROOT    = "demo_frames"
OUTPUT_ROUTING = "demo_routing.csv"
ROUTER_MODEL   = "router_model.pkl"

TARGET_FPS    = 5
MAX_DIMENSION = 256
MAX_DURATION  = 20


# ── STAGE 1: FRAME EXTRACTION ─────────────────────────────────────────────────

def get_video_dimensions(video_path):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height", "-of", "json", video_path]
    try:
        data = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
        return data["streams"][0]["width"], data["streams"][0]["height"]
    except Exception:
        return None, None


def extract_frames(video_path, video_name):
    save_dir = os.path.join(FRAMES_ROOT, video_name)
    os.makedirs(save_dir, exist_ok=True)

    width, height = get_video_dimensions(video_path)
    if width is None:
        print(f"  [skip] could not probe {video_name}")
        return False

    filters = [f"fps={TARGET_FPS}"]
    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        filters.append(f"scale={MAX_DIMENSION}:{MAX_DIMENSION}:force_original_aspect_ratio=decrease")

    cmd = ["ffmpeg", "-t", str(MAX_DURATION), "-i", video_path,
           "-vf", ",".join(filters), "-q:v", "2", "-loglevel", "error", "-y",
           os.path.join(save_dir, "frame_%04d.jpg")]
    try:
        subprocess.run(cmd, check=True)
        n_frames = len([f for f in os.listdir(save_dir) if f.endswith(".jpg")])
        print(f"  Extracted {n_frames} frames  ({width}x{height})  ->  {save_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [ffmpeg error] {e}")
        return False


# ── STAGE 2: QUALITY ASSESSMENT ───────────────────────────────────────────────

def assess_quality(video_path):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,r_frame_rate,codec_name",
           "-show_entries", "format=bit_rate,duration",
           "-of", "json", video_path]
    try:
        data   = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
        stream = data["streams"][0]
        width  = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
        codec  = stream.get("codec_name", "unknown")
        fps_str = stream.get("r_frame_rate", "30/1")
        n, d   = fps_str.split("/") if "/" in fps_str else (fps_str, "1")
        fps    = float(n) / float(d) if float(d) > 0 else 30.0
        fmt    = data["format"]
        if "bit_rate" in fmt:
            bitrate = int(fmt["bit_rate"])
        else:
            duration = float(fmt.get("duration", 0))
            bitrate  = (os.path.getsize(video_path) * 8) / duration if duration > 0 else 0
        pixel_count = width * height
        bpp = bitrate / (pixel_count * fps) if pixel_count * fps > 0 else 0
        return {"width": width, "height": height, "bitrate": bitrate,
                "fps": fps, "bpp": bpp, "codec": codec}
    except Exception as e:
        print(f"  [ffprobe error] {e}")
        return None


# ── STAGE 3: ROUTING ──────────────────────────────────────────────────────────

def route(stats):
    if not os.path.exists(ROUTER_MODEL):
        raise FileNotFoundError(f"Router model not found: '{ROUTER_MODEL}'. Run router.py first.")
    bundle   = joblib.load(ROUTER_MODEL)
    clf      = bundle["model"]
    features = bundle["features"]

    log_bitrate  = np.log1p(stats["bitrate"])
    log_bpp      = np.log1p(stats["bpp"])
    info_density = stats["height"] * log_bitrate
    row = {
        "height":       stats["height"],
        "bitrate":      stats["bitrate"],
        "bpp":          stats["bpp"],
        "log_bitrate":  log_bitrate,
        "log_bpp":      log_bpp,
        "info_density": info_density,
    }
    X = np.array([[row[f] for f in features]])
    pred_class = int(clf.predict(X)[0])
    pred_prob  = float(clf.predict_proba(X)[0, 1])
    branch     = "Branch A" if pred_class == 1 else "Branch B"
    return pred_class, pred_prob, branch


# ── STAGE 4: SAVE TO CSV ──────────────────────────────────────────────────────

def save_routing(video_name, video_path, stats, pred_class, pred_prob, branch):
    record = {
        "video_name":      video_name,
        "video_path":      video_path,
        "width":           stats["width"],
        "height":          stats["height"],
        "bitrate":         stats["bitrate"],
        "fps":             stats["fps"],
        "bpp":             stats["bpp"],
        "codec":           stats["codec"],
        "pred_class":      pred_class,
        "pred_prob":       pred_prob,
        "assigned_branch": branch,
    }
    new_row = pd.DataFrame([record])
    if os.path.exists(OUTPUT_ROUTING):
        existing = pd.read_csv(OUTPUT_ROUTING)
        existing = existing[existing["video_name"] != video_name]   # remove old entry if re-run
        df = pd.concat([existing, new_row], ignore_index=True)
    else:
        df = new_row
    df.to_csv(OUTPUT_ROUTING, index=False)
    return record


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python demorouter.py <path/to/video.mp4>")
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.isfile(video_path):
        print(f"Error: file not found — {video_path}")
        sys.exit(1)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    start = time.time()

    print(f"\n{'='*50}")
    print(f"  DEMO ROUTER  —  {video_name}")
    print(f"{'='*50}")

    # Stage 1
    print("\n[Stage 1] Extracting frames...")
    ok = extract_frames(video_path, video_name)
    if not ok:
        sys.exit(1)

    # Stage 2
    print("\n[Stage 2] Assessing quality...")
    stats = assess_quality(video_path)
    if stats is None:
        sys.exit(1)
    print(f"  Resolution : {stats['width']}x{stats['height']}")
    print(f"  Bitrate    : {stats['bitrate']:,} bps")
    print(f"  BPP        : {stats['bpp']:.6f}")
    print(f"  FPS        : {stats['fps']:.2f}")
    print(f"  Codec      : {stats['codec']}")

    # Stage 3
    print("\n[Stage 3] Routing...")
    pred_class, pred_prob, branch = route(stats)
    print(f"  Prediction : {branch}  (conf={pred_prob:.4f})")

    # Stage 4
    save_routing(video_name, video_path, stats, pred_class, pred_prob, branch)
    print(f"\n[Stage 4] Saved routing to {OUTPUT_ROUTING}")

    print(f"\n{'='*50}")
    print(f"  RESULT: {video_name}  ->  {branch}")
    print(f"  Confidence : {pred_prob:.4f}")
    print(f"  Frames     : demo_frames/{video_name}/")
    print(f"  Time       : {time.time()-start:.2f}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
