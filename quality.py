import subprocess
import os
import glob
import json
import pandas as pd
import time

# CONFIGURATION
INPUT_ROOT = "DevDataset"
OUTPUT_METADATA = "video_quality.csv"

def get_video_technical_stats(video_path):
    """
    Extracts Width, Height, Bitrate, and FPS to calculate Quality Density.
    """
    
    cmd = [
        "ffprobe", 
        "-v", "error", 
        "-select_streams", "v:0", 
        "-show_entries", "stream=width,height,r_frame_rate,codec_name", 
        "-show_entries", "format=bit_rate,duration", 
        "-of", "json", 
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        # 1. Stream Info (Resolution & FPS)
        stream = data["streams"][0]
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
        codec = stream.get("codec_name", "unknown")
        
        fps_str = stream.get("r_frame_rate", "30/1")
        if "/" in fps_str:
            n, d = fps_str.split("/")
            fps = float(n) / float(d) if float(d) > 0 else 30.0
        else:
            fps = float(fps_str)

        # 2. Format Info (Bitrate)
        fmt = data["format"]
        if "bit_rate" in fmt:
            bitrate = int(fmt["bit_rate"])
        else:
            # Fallback: Size / Duration
            duration = float(fmt.get("duration", 0))
            size = os.path.getsize(video_path)
            bitrate = (size * 8) / duration if duration > 0 else 0

        return width, height, bitrate, fps, codec

    except Exception as e:
        print(f"Error probing {os.path.basename(video_path)}: {e}")
        return 0, 0, 0, 0, "error"

def main():
    start = time.time()
    data = []
    
    search_paths = [
        (os.path.join("Real", "high_quality_real"), "real"),
        (os.path.join("Real", "low_quality_real"),  "real"),
        (os.path.join("Ai",   "high_quality_ai"),   "ai"),
        (os.path.join("Ai",   "low_quality_ai"),    "ai")
    ]
    
    print(f"Scanning {INPUT_ROOT} for metadata...")
    
    for relative_path, label in search_paths:
        full_path = os.path.join(INPUT_ROOT, relative_path)
        subfolder_name = os.path.basename(relative_path)
        
        if not os.path.exists(full_path):
            continue
            
        videos = glob.glob(os.path.join(full_path, "*.mp4"))
        print(f"Processing {len(videos)} videos in '{subfolder_name}'...")
        
        for video_path in videos:
            w, h, br, fps, codec = get_video_technical_stats(video_path)
            
            if w == 0: continue
            
            # Calculate Bits Per Pixel (BPP)
            # Standard Formula: Bitrate / (Pixels * FPS)
            # Typical H.264 Quality:
            #   0.10+ = High Quality
            #   0.05- = Blocky/Low Quality
            pixel_count = w * h
            bpp = br / (pixel_count * fps) if (pixel_count * fps) > 0 else 0
            
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            
            data.append({
                "video_name": video_name,
                "label": label,
                "subfolder": subfolder_name,
                "width": w,
                "height": h,
                "bitrate": br,
                "fps": fps,
                "bpp": bpp, 
                "codec": codec
            })

    if data:
        df = pd.DataFrame(data)
        df.to_csv(OUTPUT_METADATA, index=False)
        print(f"\nSaved metadata to {OUTPUT_METADATA}")
        print("\n--- STATS BY SUBFOLDER ---")
        print(df.groupby("subfolder")[['height', 'bitrate', 'bpp']].mean())
        end=time.time()
        print(f"\nMetadata extraction completed in {end-start:.2f} seconds.")
    else:
        print("No videos found.")

if __name__ == "__main__":
    main()