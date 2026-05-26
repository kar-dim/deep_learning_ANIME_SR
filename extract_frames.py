# Dimitris Karatzas aivc25007

from pathlib import Path
import subprocess
import sys

INPUT_DIR = Path(".")
OUTPUT_DIR = Path("dataset_frames")

# Extract one frame every N seconds
INTERVAL_SECONDS = 60
# Output height
TARGET_HEIGHT = 720
# Scaling algorithm
SCALER = "lanczos"
# Supported formats
VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".mov", ".webm",".flv"}

def find_videos(directory: Path):
    for path in directory.rglob("*"):
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path

def process_video(video_path: Path):
    """Extract one frame every INTERVAL_SECONDS from video_path, scaled to TARGET_HEIGHT, saved as PNG."""
    relative_path = video_path.relative_to(INPUT_DIR)
    episode_name = relative_path.with_suffix("")
    output_folder = OUTPUT_DIR / episode_name
    output_folder.mkdir(parents=True, exist_ok=True)

    # Skip if already processed (any PNG present means a previous run completed this video)
    existing_frames = list(output_folder.glob("*.png"))
    if existing_frames:
        print(f"SKIP: {video_path}")
        return

    output_pattern = output_folder / "frame_%04d.png"

    # fps=1/N -> one frame every N seconds, scale=-1:H -> keeps aspect ratio at height H
    vf = f"fps=1/{INTERVAL_SECONDS},scale=-1:{TARGET_HEIGHT}:flags={SCALER}"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-stats",
        "-i", str(video_path),
        "-vf", vf,
        "-compression_level", "1", # level 1 = least compression, fastest PNG encode
        "-vsync", "vfr", # variable framerate to avoid duplicate frames
        str(output_pattern)
    ]

    print(f"\nPROCESSING: {video_path}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"FAILED: {video_path}")
    else:
        print(f"DONE: {video_path}")


def main():
    if not INPUT_DIR.exists():
        print(f"Input directory not found: {INPUT_DIR}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    videos = list(find_videos(INPUT_DIR))

    if not videos:
        print("No videos found.")
        return

    print(f"Found {len(videos)} video(s).\n")

    for video in videos:
        process_video(video)

    print("\nAll processing complete.")


if __name__ == "__main__":
    main()
