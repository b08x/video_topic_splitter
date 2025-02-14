# test_moviepy.py
from moviepy.editor import VideoFileClip

video_path = "/home/b08x/Videos/temp/yt_4o9093zuRiY_20250213_234025/source_video.mp4"  # Use the actual path to your downloaded video

try:
    video = VideoFileClip(video_path)
    fps = video.fps
    print(f"MoviePy loaded video successfully.")
    print(f"Video FPS: {fps}")
    video.close()
except Exception as e:
    print(f"MoviePy failed to load video or get FPS.")
    print(f"Error details: {e}")