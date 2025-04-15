"""
Video processing utilities for scene detection and segmentation.
"""

import os
import csv
import logging
import subprocess
from typing import List, Tuple, Optional
import tempfile
import json
from pathlib import Path

logger = logging.getLogger(__name__)

def read_scenes_from_csv(csv_path: str) -> List[Tuple[float, float]]:
    """
    Read scene boundaries from a CSV file created by PySceneDetect.
    
    Args:
        csv_path: Path to the CSV file containing scene information
        
    Returns:
        List of tuples containing (start_time, end_time) in seconds
    """
    scene_boundaries = []
    
    try:
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            # Skip header row
            next(reader, None)
            
            for row in reader:
                try:
                    # PySceneDetect CSV format: Scene Number,Start Time (seconds),Start Frame,End Time (seconds),End Frame
                    if len(row) >= 5:
                        start_time = float(row[1])  # Start Time in seconds
                        end_time = float(row[3])    # End Time in seconds
                        scene_boundaries.append((start_time, end_time))
                    else:
                        logger.warning(f"Skipping row with insufficient data: {row}")
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing row {row}: {e}")
                    continue
                    
        logger.info(f"Successfully read {len(scene_boundaries)} scene boundaries from {csv_path}")
        return scene_boundaries
        
    except Exception as e:
        logger.error(f"Failed to read scene boundaries from CSV {csv_path}: {e}", exc_info=True)
        raise

def detect_scenes(video_path: str, output_dir: str) -> List[Tuple[float, float]]:
    """
    Detect scene changes in a video using PySceneDetect.
    
    Args:
        video_path: Path to the input video file
        output_dir: Directory to save PySceneDetect output files
        
    Returns:
        List of tuples containing (start_time, end_time) in seconds
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare output paths
        csv_path = os.path.join(output_dir, "scenes.csv")
        
        # Run PySceneDetect command
        # Using content detector with threshold 30
        cmd = [
            "scenedetect",
            "-i", video_path,
            "detect-content",
            "-t", "30",  # Threshold
            "list-scenes",
            "-o", csv_path
        ]
        
        logger.info(f"Running PySceneDetect with command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"PySceneDetect failed: {result.stderr}")
            raise RuntimeError(f"PySceneDetect command failed: {result.stderr}")
            
        # Read the generated CSV file
        if os.path.exists(csv_path):
            scene_boundaries = read_scenes_from_csv(csv_path)
            return scene_boundaries
        else:
            logger.error(f"PySceneDetect did not generate the expected CSV file at {csv_path}")
            raise FileNotFoundError(f"Expected CSV file not found at {csv_path}")
            
    except Exception as e:
        logger.error(f"Error in scene detection: {e}", exc_info=True)
        raise

def align_timestamps_to_keyframes(video_path: str, timestamps: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Align segment timestamps to nearby keyframes for cleaner cuts.
    
    Args:
        video_path: Path to the input video file
        timestamps: List of (start_time, end_time) tuples in seconds
        
    Returns:
        List of aligned (start_time, end_time) tuples
    """
    try:
        # Extract keyframe information using ffprobe
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
            keyframes_json = temp_file.name
            
        cmd = [
            "ffprobe", 
            "-v", "quiet",
            "-select_streams", "v:0",
            "-show_frames", 
            "-show_entries", "frame=pkt_pts_time,key_frame",
            "-of", "json",
            video_path
        ]
        
        logger.info(f"Extracting keyframe information with command: {' '.join(cmd)}")
        with open(keyframes_json, 'w') as f:
            subprocess.run(cmd, stdout=f)
            
        # Read the keyframe data
        with open(keyframes_json, 'r') as f:
            frames_data = json.load(f)
            
        # Clean up temp file
        os.unlink(keyframes_json)
        
        # Extract keyframe timestamps
        keyframe_times = []
        for frame in frames_data.get('frames', []):
            if frame.get('key_frame') == 1 and 'pkt_pts_time' in frame:
                try:
                    keyframe_times.append(float(frame['pkt_pts_time']))
                except (ValueError, TypeError):
                    continue
                    
        if not keyframe_times:
            logger.warning("No keyframe information found, returning original timestamps")
            return timestamps
            
        # Sort keyframe times
        keyframe_times.sort()
        
        # Align each timestamp to the nearest keyframe
        aligned_timestamps = []
        for start_time, end_time in timestamps:
            # Find nearest keyframe for start time (prefer earlier keyframe)
            aligned_start = start_time
            for kf_time in keyframe_times:
                if kf_time <= start_time:
                    aligned_start = kf_time
                else:
                    break
                    
            # Find nearest keyframe for end time (prefer later keyframe)
            aligned_end = end_time
            for kf_time in reversed(keyframe_times):
                if kf_time >= end_time:
                    aligned_end = kf_time
                else:
                    break
                    
            # Ensure we have a valid segment
            if aligned_end <= aligned_start:
                logger.warning(f"Invalid aligned segment: {aligned_start}-{aligned_end}, using original")
                aligned_timestamps.append((start_time, end_time))
            else:
                aligned_timestamps.append((aligned_start, aligned_end))
                
        logger.info(f"Aligned {len(timestamps)} timestamps to keyframes")
        return aligned_timestamps
        
    except Exception as e:
        logger.error(f"Error aligning timestamps to keyframes: {e}", exc_info=True)
        logger.warning("Using original timestamps due to alignment error")
        return timestamps

def segment_video(
    video_path: str, 
    output_dir: str, 
    timestamps: List[Tuple[float, float]], 
    output_name_template: str = "segment_$INDEX.mp4"
) -> List[str]:
    """
    Segment a video into multiple clips based on the provided timestamps.
    
    Args:
        video_path: Path to the input video file
        output_dir: Directory to save the segmented video files
        timestamps: List of (start_time, end_time) tuples in seconds
        output_name_template: Template for output filenames, $INDEX will be replaced with segment number
        
    Returns:
        List of paths to the created segment files
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        segment_paths = []
        for i, (start_time, end_time) in enumerate(timestamps, 1):
            # Create output filename
            output_filename = output_name_template.replace("$INDEX", f"{i:03d}")
            output_path = os.path.join(output_dir, output_filename)
            
            # Calculate duration
            duration = end_time - start_time
            
            # Skip very short segments
            if duration < 0.5:  # Less than half a second
                logger.warning(f"Skipping very short segment {i}: {start_time}-{end_time} ({duration:.2f}s)")
                continue
                
            # Use ffmpeg to extract the segment
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output files
                "-ss", str(start_time),  # Start time
                "-i", video_path,  # Input file
                "-t", str(duration),  # Duration
                "-c", "copy",  # Copy codecs (fast)
                "-avoid_negative_ts", "1",
                output_path
            ]
            
            logger.info(f"Creating segment {i} with command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Failed to create segment {i}: {result.stderr}")
                # Try again with re-encoding (slower but more reliable)
                logger.info(f"Retrying segment {i} with re-encoding")
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-ss", str(start_time),
                    "-i", video_path,
                    "-t", str(duration),
                    "-c:v", "libx264",  # Re-encode video
                    "-c:a", "aac",      # Re-encode audio
                    "-preset", "fast",
                    output_path
                ]
                retry_result = subprocess.run(cmd, capture_output=True, text=True)
                
                if retry_result.returncode != 0:
                    logger.error(f"Failed to create segment {i} even with re-encoding: {retry_result.stderr}")
                    continue
            
            # Verify the segment was created
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                segment_paths.append(output_path)
                logger.info(f"Successfully created segment {i}: {output_path}")
            else:
                logger.error(f"Segment file {output_path} was not created or is empty")
                
        logger.info(f"Created {len(segment_paths)} video segments")
        return segment_paths
        
    except Exception as e:
        logger.error(f"Error segmenting video: {e}", exc_info=True)
        return []