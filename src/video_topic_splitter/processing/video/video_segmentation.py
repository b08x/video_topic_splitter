#!/usr/bin/env python3
"""
Video segmentation functionality for creating topic-based video segments.
Adapted from vtsV3 with improvements for the new project structure.
"""

import logging
import os
import subprocess
from typing import List, Tuple, Optional, Dict, Any
from ...progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)


def segment_video_by_topics(
    video_path: str,
    topic_segments: List[Dict[str, Any]],
    output_structure,
    progress_tracker: ProgressTracker = None
) -> List[Dict[str, Any]]:
    """
    Segment a video based on topic segments with timestamps.
    
    Args:
        video_path: Path to the input video file
        topic_segments: List of topic segments with timing information
        output_structure: ProjectStructure instance for managing paths
        progress_tracker: Optional progress tracker
        
    Returns:
        List of segment information with file paths
    """
    logger.info(f"Starting video segmentation for {len(topic_segments)} segments")
    
    if progress_tracker:
        progress_tracker.start_phase("Video Segmentation")
    
    # Extract timestamps from topic segments
    timestamps = []
    for segment in topic_segments:
        start_time = segment.get("start", 0)
        end_time = segment.get("end", 0)
        timestamps.append((start_time, end_time))
    
    if progress_tracker:
        progress_tracker.update_phase_progress(10.0, "Preparing video segmentation...")
    
    # Align timestamps to keyframes for better accuracy
    aligned_timestamps = align_timestamps_to_keyframes(video_path, timestamps)
    
    if progress_tracker:
        progress_tracker.update_phase_progress(20.0, "Timestamps aligned to keyframes")
    
    # Create video segments
    segmented_files = []
    total_segments = len(topic_segments)
    
    for i, (segment_data, (start_time, end_time)) in enumerate(zip(topic_segments, aligned_timestamps)):
        segment_num = i + 1
        topic_name = segment_data.get("topic", f"Segment_{segment_num}")
        
        if progress_tracker:
            progress_tracker.update_phase_progress(
                20.0 + (i / total_segments) * 60.0,
                f"Creating segment {segment_num}/{total_segments}: {topic_name}"
            )
        
        # Get segment paths
        segment_paths = output_structure.get_segment_paths(segment_num, topic_name)
        
        # Create video segment
        video_created = create_video_segment(
            video_path,
            segment_paths["video_file"],
            start_time,
            end_time,
            segment_num
        )
        
        # Create audio segment
        audio_created = create_audio_segment(
            video_path,
            segment_paths["audio_file"],
            start_time,
            end_time,
            segment_num
        )
        
        # Add segment information
        segment_info = {
            "segment_number": segment_num,
            "topic": topic_name,
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "paths": segment_paths,
            "video_created": video_created,
            "audio_created": audio_created,
            "original_segment_data": segment_data
        }
        
        segmented_files.append(segment_info)
    
    if progress_tracker:
        progress_tracker.update_phase_progress(90.0, "Video segmentation complete")
        progress_tracker.complete_phase("Video Segmentation")
    
    logger.info(f"Successfully created {len(segmented_files)} video segments")
    return segmented_files


def create_video_segment(
    video_path: str,
    output_path: str,
    start_time: float,
    end_time: float,
    segment_num: int
) -> bool:
    """
    Create a video segment using ffmpeg.
    
    Args:
        video_path: Path to input video
        output_path: Path for output segment
        start_time: Start time in seconds
        end_time: End time in seconds
        segment_num: Segment number for logging
        
    Returns:
        True if successful, False otherwise
    """
    try:
        duration = end_time - start_time
        if duration <= 0:
            logger.warning(f"Skipping segment {segment_num} with invalid duration")
            return False
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # FFmpeg command for video segmentation
        command = [
            "ffmpeg",
            "-ss", str(start_time),
            "-i", video_path,
            "-t", str(duration),
            "-c:v", "copy",  # Copy video codec (fastest)
            "-c:a", "copy",  # Copy audio codec (fastest)
            "-avoid_negative_ts", "make_zero",
            "-y",  # Overwrite output file
            output_path
        ]
        
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"Video segment {segment_num} created: {output_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error creating video segment {segment_num}: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error creating video segment {segment_num}: {e}")
        return False


def create_audio_segment(
    video_path: str,
    output_path: str,
    start_time: float,
    end_time: float,
    segment_num: int
) -> bool:
    """
    Create an audio segment from video using ffmpeg.
    
    Args:
        video_path: Path to input video
        output_path: Path for output audio segment
        start_time: Start time in seconds
        end_time: End time in seconds
        segment_num: Segment number for logging
        
    Returns:
        True if successful, False otherwise
    """
    try:
        duration = end_time - start_time
        if duration <= 0:
            logger.warning(f"Skipping audio segment {segment_num} with invalid duration")
            return False
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # FFmpeg command for audio extraction
        command = [
            "ffmpeg",
            "-i", video_path,
            "-ss", str(start_time),
            "-t", str(duration),
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # 16-bit PCM
            "-ar", "22050",  # 22kHz sample rate
            "-ac", "1",  # Mono
            "-y",  # Overwrite output file
            output_path
        ]
        
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"Audio segment {segment_num} created: {output_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error creating audio segment {segment_num}: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error creating audio segment {segment_num}: {e}")
        return False


def align_timestamps_to_keyframes(
    video_path: str, 
    timestamps: List[Tuple[float, float]], 
    tolerance: float = 0.1
) -> List[Tuple[float, float]]:
    """
    Align timestamps to nearest keyframes for accurate segmentation.
    
    Args:
        video_path: Path to the input video file
        timestamps: List of (start_time, end_time) tuples
        tolerance: Maximum allowed time difference for adjustment
        
    Returns:
        List of adjusted (start_time, end_time) tuples
    """
    try:
        # Get keyframe information using ffprobe
        ffprobe_command = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-skip_frame", "nokey",
            "-show_entries", "frame=pkt_pts_time,pict_type",
            "-of", "csv=p=0",
            video_path
        ]
        
        result = subprocess.run(ffprobe_command, check=True, capture_output=True, text=True)
        
        # Parse keyframe times (I-frames)
        keyframe_times = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(',')
                if len(parts) >= 2 and parts[1] == 'I':
                    keyframe_times.append(float(parts[0]))
        
        # Align timestamps to keyframes
        adjusted_timestamps = []
        for start_time, end_time in timestamps:
            adjusted_start = _find_nearest_keyframe(keyframe_times, start_time, tolerance)
            adjusted_end = _find_nearest_keyframe(keyframe_times, end_time, tolerance)
            adjusted_timestamps.append((adjusted_start, adjusted_end))
        
        logger.info(f"Aligned {len(timestamps)} timestamps to keyframes")
        return adjusted_timestamps
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error getting keyframe info: {e.stderr}")
        return timestamps  # Return original timestamps if alignment fails
    except Exception as e:
        logger.error(f"Unexpected error in keyframe alignment: {e}")
        return timestamps


def _find_nearest_keyframe(
    keyframe_times: List[float], 
    target_time: float, 
    tolerance: float
) -> float:
    """
    Find the nearest keyframe time to the target time within tolerance.
    
    Args:
        keyframe_times: List of keyframe times
        target_time: Target time to find nearest keyframe for
        tolerance: Maximum allowed time difference
        
    Returns:
        Nearest keyframe time or original target_time if none found
    """
    if not keyframe_times:
        return target_time
    
    # Find the keyframe with minimum distance to target
    nearest_time = min(keyframe_times, key=lambda x: abs(x - target_time))
    
    # Only use keyframe if within tolerance
    if abs(nearest_time - target_time) <= tolerance:
        return nearest_time
    
    return target_time


def extract_frames_at_timestamps(
    video_path: str,
    timestamps: List[float],
    output_dir: str,
    output_template: str = "frame_%04d.jpg",
    format: str = "jpg",
    quality: int = 90
) -> List[str]:
    """
    Extract frames from a video at specific timestamps using enhanced dual-method approach.
    
    Args:
        video_path: Path to the video file
        timestamps: List of timestamps in seconds
        output_dir: Directory to save extracted frames
        output_template: Filename template for output frames
        format: Output image format (jpg, png)
        quality: Compression quality for JPEG (0-100)
        
    Returns:
        List of successfully extracted frame paths
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return []
        
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
) -> List[str]:
    """Extract frames using FFmpeg (more efficient for specific timestamps)."""
    frame_paths = []
    
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
            if os.path.exists(output_path):
                frame_paths.append(output_path)
                logger.debug(f"Extracted frame at {timestamp:.1f}s: {output_path}")
            else:
                logger.warning(f"FFmpeg did not produce output file at {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error for timestamp {timestamp}: {e.stderr.decode() if e.stderr else str(e)}")
            
    return frame_paths


def _extract_frames_opencv(
    video_path: str,
    timestamps: List[float],
    output_dir: str,
    output_template: str,
    format: str,
    quality: int
) -> List[str]:
    """Extract frames using OpenCV (fallback method)."""
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return []
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        logger.warning(f"Invalid FPS value: {fps}. Using default of 30.")
        fps = 30
        
    frame_paths = []
    
    for i, timestamp in enumerate(timestamps):
        # Convert timestamp to frame number
        frame_num = int(timestamp * fps)
        
        # Set position to the frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Failed to read frame at timestamp {timestamp}s")
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
        success = cv2.imwrite(output_path, frame, params)
        
        if success and os.path.exists(output_path):
            frame_paths.append(output_path)
            logger.debug(f"Extracted frame at {timestamp:.1f}s: {output_path}")
        else:
            logger.warning(f"Failed to save frame to {output_path}")
            
    cap.release()
    return frame_paths


def extract_scene_keyframes(
    video_path: str,
    scene_boundaries: List[Tuple[float, float]],
    output_dir: str,
    frames_per_scene: int = 1,
    format: str = "jpg",
    quality: int = 90
) -> Dict[int, List[str]]:
    """
    Extract representative keyframes from each scene using enhanced extraction.
    
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
        
        # Extract frames using enhanced method
        frame_paths = extract_frames_at_timestamps(
            video_path=video_path,
            timestamps=timestamps,
            output_dir=scene_dir,
            output_template=f"frame_%04d.{format}",
            format=format,
            quality=quality
        )
        
        if frame_paths:
            scene_frames[i] = frame_paths
        
    return scene_frames


def extract_segment_frames(
    video_path: str,
    frames_dir: str,
    start_time: float,
    end_time: float,
    num_frames: int = 5,
    format: str = "jpg",
    quality: int = 90
) -> List[str]:
    """
    Extract frames from a video segment for visual analysis (enhanced version).
    
    Args:
        video_path: Path to input video
        frames_dir: Directory to save frames
        start_time: Segment start time
        end_time: Segment end time
        num_frames: Number of frames to extract
        format: Output image format (jpg, png)
        quality: Compression quality for JPEG (0-100)
        
    Returns:
        List of frame file paths
    """
    try:
        os.makedirs(frames_dir, exist_ok=True)
        
        duration = end_time - start_time
        if duration <= 0:
            logger.warning("Invalid segment duration for frame extraction")
            return []
        
        # Calculate frame extraction times
        if num_frames == 1:
            timestamps = [start_time + duration / 2]  # Middle frame
        else:
            timestamps = [
                start_time + (i * duration / (num_frames - 1))
                for i in range(num_frames)
            ]
        
        # Use enhanced extraction method
        frame_paths = extract_frames_at_timestamps(
            video_path=video_path,
            timestamps=timestamps,
            output_dir=frames_dir,
            output_template=f"frame_%04d.{format}",
            format=format,
            quality=quality
        )
        
        logger.info(f"Extracted {len(frame_paths)} frames from segment")
        return frame_paths
        
    except Exception as e:
        logger.error(f"Unexpected error in frame extraction: {e}")
        return []


def validate_segment_files(segment_info: Dict[str, Any]) -> Dict[str, bool]:
    """
    Validate that segment files were created successfully.
    
    Args:
        segment_info: Segment information dictionary
        
    Returns:
        Dictionary with validation results
    """
    validation = {
        "video_exists": False,
        "audio_exists": False,
        "video_size": 0,
        "audio_size": 0,
        "all_valid": False
    }
    
    try:
        paths = segment_info.get("paths", {})
        
        # Check video file
        video_path = paths.get("video_file")
        if video_path and os.path.exists(video_path):
            validation["video_exists"] = True
            validation["video_size"] = os.path.getsize(video_path)
        
        # Check audio file
        audio_path = paths.get("audio_file")
        if audio_path and os.path.exists(audio_path):
            validation["audio_exists"] = True
            validation["audio_size"] = os.path.getsize(audio_path)
        
        # Check if both files exist and have reasonable sizes
        validation["all_valid"] = (
            validation["video_exists"] and 
            validation["audio_exists"] and
            validation["video_size"] > 1000 and  # At least 1KB
            validation["audio_size"] > 1000
        )
        
    except Exception as e:
        logger.error(f"Error validating segment files: {e}")
    
    return validation