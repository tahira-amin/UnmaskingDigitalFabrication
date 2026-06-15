import subprocess
import os
import glob
import json
import time

# CONFIGURATION
INPUT_ROOT = "DevDataset"
OUTPUT_ROOT = "processed"
TARGET_FPS = 5
MAX_DIMENSION = 256
MAX_DURATION = 20  

def get_video_info(video_path):
    """
    Uses ffprobe to get width and height without decoding the video.
    Returns (width, height)
    """
    cmd = [
        "ffprobe", 
        "-v", "error", 
        "-select_streams", "v:0", 
        "-show_entries", "stream=width,height", 
        "-of", "json", 
        video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        width = data["streams"][0]["width"]
        height = data["streams"][0]["height"]
        return width, height
    except Exception as e:
        print(f"Error probing {video_path}: {e}")
        return None, None

def process_video_ffmpeg(video_path, relative_path):
    filename = os.path.basename(video_path)
    video_name, _ = os.path.splitext(filename)
    
    # Define output directory to mirror input structure
    # e.g. processed/Real/high_quality_real/video_name
    save_dir = os.path.join(OUTPUT_ROOT, relative_path, video_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Get dimensions first
    width, height = get_video_info(video_path)
    if width is None: 
        print(f"Skipping {filename} - could not get video info")
        return

    # 2. Construct the Filter String
    # Start with defining FPS
    filters = [f"fps={TARGET_FPS}"]
    
    # Scale to fit within MAX_DIMENSION while preserving aspect ratio
    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        filters.append(f"scale={MAX_DIMENSION}:{MAX_DIMENSION}:force_original_aspect_ratio=decrease")
    
    filter_string = ",".join(filters)

    # 3. Build FFmpeg Command
    output_pattern = os.path.join(save_dir, "frame_%04d.jpg")
    
    cmd = [
        "ffmpeg",
        "-t", str(MAX_DURATION), # Stop processing after 20s
        "-i", video_path,
        "-vf", filter_string,
        "-q:v", "2",           # High Quality JPG
        "-loglevel", "error",  # Keep terminal clean
        "-y",                  # Overwrite if exists
        output_pattern
    ]

    print(f"Processing ({relative_path}): {filename} ({width}x{height})")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error on {filename}: {e}")

def main():
    start_time = time.time()
    
    # UPDATED SEARCH STRUCTURE
    # Now we map the specific input pattern to a specific output subfolder path
    # output structure: processed/Parent/Subfolder/VideoName
    search_paths = [
        # (Path Pattern, Output Relative Path)
        (os.path.join(INPUT_ROOT, "Real", "high_quality_real", "*.mp4"), os.path.join("Real", "high_quality_real")),
        (os.path.join(INPUT_ROOT, "Real", "low_quality_real", "*.mp4"),  os.path.join("Real", "low_quality_real")),
        (os.path.join(INPUT_ROOT, "Ai", "high_quality_ai", "*.mp4"),     os.path.join("Ai", "high_quality_ai")),
        (os.path.join(INPUT_ROOT, "Ai", "low_quality_ai", "*.mp4"),      os.path.join("Ai", "low_quality_ai"))
    ]

    total_videos = 0

    for pattern, rel_path in search_paths:
        videos = glob.glob(pattern)
        print(f"\nScanning {pattern}...")
        print(f"Found {len(videos)} videos. Output target: '{rel_path}'")
        
        for vid in videos:
            process_video_ffmpeg(vid, rel_path)
            total_videos += 1

    end_time = time.time()    
    elapsed_time = end_time - start_time
    print(f"\n" + "="*40)
    print(f"Extraction Complete!")
    print(f"Processed {total_videos} videos.")
    print(f"Output Structure: {OUTPUT_ROOT}/[Real/Ai]/[Quality]/[VideoName]")
    print(f"Total Execution Time: {elapsed_time:.2f} seconds")
    print(f"="*40)

if __name__ == "__main__":
    main()