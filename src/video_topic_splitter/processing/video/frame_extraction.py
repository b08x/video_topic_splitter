"""Video frame extraction utilities."""

import logging
import os
import subprocess
from typing import List, Optional, Union

import cv2

logger = logging.getLogger(__name__)

def extract_frames_at_timestamps(
    video_path: str,
    timestamps: List[float],
    output_dir: str,
    output_template: str = "frame_%04d.jpg",
    format: str = "jpg",
    quality: int = 90
) -> bool:
    """
    Extract frames from a video at specific timestamps.
    
    Args:
        video_path: Path to the video file
        timestamps: List of timestamps in seconds
        output_dir: Directory to save extracted frames
        output_template: Filename template for output frames
        format: Output image format (jpg, png)
        quality: Compression quality for JPEG (0-100)
        
    Returns:
        True if extraction was successful, False otherwise
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return False
        
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Try using FFmpeg for more efficient extraction
        return _extract_frames_ffmpeg(
            video_path, timestamps, output_dir, output_template, format, quality
        )
    except Exception as e:
        logger.warning(f"FFmpeg extraction failed: {e}. Falling back to OpenCV.")
        return _extract_frames_opencv(
            video_path, timestamps, output_dir, output_template, format, quality
        )

def _extract_frames_ffmpeg(
    video_path: str,
    timestamps: List[float],
    output_dir: str,
    output_template: str,
    format: str,
    quality: int
) -> bool:
    """Extract frames using FFmpeg (more efficient for specific timestamps)."""
    success = True
    
    for i, timestamp in enumerate(timestamps):
        output_path = os.path.join(output_dir, output_template.replace('%04d', f'{i:04d}'))
        
        # Ensure the output path has the correct extension
        if not output_path.lower().endswith(f'.{format.lower()}'):
            base, _ = os.path.splitext(output_path)
            output_path = f"{base}.{format.lower()}"
            
        # Build FFmpeg command
        cmd = [
            'ffmpeg',
            '-ss', str(timestamp),  # Seek to timestamp
            '-i', video_path,       # Input file
            '-frames:v', '1',       # Extract one frame
            '-q:v', str(min(31, 31 - (quality // 3))),  # Quality (FFmpeg scale: 2-31, lower is better)
            '-y',                   # Overwrite output
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            if not os.path.exists(output_path):
                logger.warning(f"FFmpeg did not produce output file at {output_path}")
                success = False
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
            success = False
            
    return success

def _extract_frames_opencv(
    video_path: str,
    timestamps: List[float],
    output_dir: str,
    output_template: str,
    format: str,
    quality: int
) -> bool:
    """Extract frames using OpenCV (fallback method)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return False
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        logger.warning(f"Invalid FPS value: {fps}. Using default of 30.")
        fps = 30
        
    success = True
    
    for i, timestamp in enumerate(timestamps):
        # Convert timestamp to frame number
        frame_num = int(timestamp * fps)
        
        # Set position to the frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Failed to read frame at timestamp {timestamp}s")
            success = False
            continue
            
        # Save the frame
        output_path = os.path.join(output_dir, output_template.replace('%04d', f'{i:04d}'))
        
        # Ensure the output path has the correct extension
        if not output_path.lower().endswith(f'.{format.lower()}'):
            base, _ = os.path.splitext(output_path)
            output_path = f"{base}.{format.lower()}"
            
        # Set compression parameters
        if format.lower() == 'jpg' or format.lower() == 'jpeg':
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif format.lower() == 'png':
            params = [cv2.IMWRITE_PNG_COMPRESSION, min(9, 9 - (quality // 10))]
        else:
            params = []
            
        # Save the frame
        cv2.imwrite(output_path, frame, params)
        
        if not os.path.exists(output_path):
            logger.warning(f"Failed to save frame to {output_path}")
            success = False
            
    cap.release()
    return success

def extract_scene_keyframes(
    video_path: str,
    scene_boundaries: List[tuple],
    output_dir: str,
    frames_per_scene: int = 1,
    format: str = "jpg",
    quality: int = 90
) -> dict:
    """
    Extract representative keyframes from each scene.
    
    Args:
        video_path: Path to the video file
        scene_boundaries: List of (start_time, end_time) tuples in seconds
        output_dir: Directory to save extracted frames
        frames_per_scene: Number of frames to extract per scene
        format: Output image format (jpg, png)
        quality: Compression quality for JPEG (0-100)
        
    Returns:
        Dictionary mapping scene index to list of extracted frame paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    scene_frames = {}
    
    for i, (start_time, end_time) in enumerate(scene_boundaries):
        scene_duration = end_time - start_time
        scene_dir = os.path.join(output_dir, f"scene_{i}")
        os.makedirs(scene_dir, exist_ok=True)
        
        # Calculate timestamps for this scene
        if frames_per_scene == 1:
            # Just take the middle frame
            timestamps = [start_time + (scene_duration / 2)]
        else:
            # Distribute frames evenly
            timestamps = [
                start_time + (j * scene_duration / (frames_per_scene - 1))
                for j in range(frames_per_scene)
            ]
            # Ensure we don't go beyond the end time
            timestamps = [min(t, end_time - 0.1) for t in timestamps]
        
        # Extract frames
        success = extract_frames_at_timestamps(
            video_path=video_path,
            timestamps=timestamps,
            output_dir=scene_dir,
            output_template=f"frame_%04d.{format}",
            format=format,
            quality=quality
        )
        
        if success:
            # Collect frame paths
            frame_paths = [
                os.path.join(scene_dir, f"frame_{j:04d}.{format}")
                for j in range(len(timestamps))
            ]
            # Filter to only include files that exist
            frame_paths = [p for p in frame_paths if os.path.exists(p)]
            
            if frame_paths:
                scene_frames[i] = frame_paths
        
    return scene_frames