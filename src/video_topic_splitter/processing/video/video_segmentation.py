# processing/video/video_segmentation.py
#!/usr/bin/env python3
"""
Contains functions for segmenting videos based on timestamps.
"""

import logging
import os
import subprocess
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


def segment_video(
    video_path: str,
    output_dir: str,
    timestamps: List[Tuple[float, float]],
    output_name_template: str = "segment_$INDEX.mp4",
) -> List[str]:
    """
    Segments a video into multiple clips based on provided timestamps.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory to save the segmented video clips.
        timestamps: A list of tuples, where each tuple represents the start and end
                    timestamps (in seconds) of a segment.
        output_name_template: Template for naming the output video files.
                              "$INDEX" will be replaced with the segment index (1-based).

    Returns:
        A list of paths to the generated video segment files.
    """
    os.makedirs(output_dir, exist_ok=True)
    segment_paths: List[str] = []

    for i, (start_time, end_time) in enumerate(timestamps):
        output_path = os.path.join(
            output_dir, output_name_template.replace("$INDEX", str(i + 1))
        )
        try:
            # Calculate segment duration
            duration = end_time - start_time
            if duration <= 0:
                logger.warning(
                    f"Skipping segment {i + 1} with invalid duration (start={start_time}, end={end_time})"
                )
                continue  # Skip segments with invalid duration

            # Construct ffmpeg command
            command = [
                "ffmpeg",
                "-ss",
                str(start_time),
                "-i",
                video_path,
                "-t",
                str(duration),
                "-c:v",
                "copy",  # Copy video codec (fastest)
                "-c:a",
                "copy",  # Copy audio codec (fastest)
                output_path,
            ]

            subprocess.run(command, check=True, capture_output=True, text=True)
            segment_paths.append(os.path.abspath(output_path))
            logger.info(f"Segment {i + 1} saved to {output_path}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Error segmenting video: {e.stderr}")
        except Exception as e:
            logger.error(f"Unexpected error during video segmentation: {e}")

    return segment_paths


def align_timestamps_to_keyframes(
    video_path: str, timestamps: List[Tuple[float, float]], tolerance: float = 0.1
) -> List[Tuple[float, float]]:
    """
    Adjusts timestamps to the nearest keyframes in the video.

    This is important for accurate seeking and segmenting, as seeking to non-keyframes
    can result in inaccurate or corrupted video segments.

    Args:
        video_path: Path to the input video file.
        timestamps: List of (start_time, end_time) tuples.
        tolerance: Maximum allowed time difference (in seconds) for adjustment.

    Returns:
        A list of adjusted (start_time, end_time) tuples.
    """
    adjusted_timestamps: List[Tuple[float, float]] = []
    try:
        # Use ffprobe to get keyframe information
        ffprobe_command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=pkt_pts_time,pict_type",
            "-of",
            "csv=p=0",
            video_path,
        ]
        ffprobe_output = subprocess.run(
            ffprobe_command, check=True, capture_output=True, text=True
        ).stdout
        keyframe_times = [
            float(line.split(",")[0])
            for line in ffprobe_output.strip().split("\n")
            if line.split(",")[1] == "I"
        ]  # 'I' frame is a keyframe

        for start_time, end_time in timestamps:
            adjusted_start = _find_nearest_keyframe(
                keyframe_times, start_time, tolerance
            )
            adjusted_end = _find_nearest_keyframe(
                keyframe_times, end_time, tolerance
            )
            adjusted_timestamps.append((adjusted_start, adjusted_end))
        logger.info(f"Timestamps adjusted to keyframes for {video_path}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Error getting keyframe info: {e.stderr}")
        # If ffprobe fails, return original timestamps (segmentation might be inaccurate)
        return timestamps
    except Exception as e:
        logger.error(f"Unexpected error in align_timestamps_to_keyframes: {e}")
        return timestamps
    return adjusted_timestamps


def _find_nearest_keyframe(
    keyframe_times: List[float], target_time: float, tolerance: float
) -> float:
    """
    Finds the nearest keyframe time to the target time within a tolerance.

    Args:
        keyframe_times: List of keyframe times.
        target_time: The time to find the nearest keyframe to.
        tolerance: Maximum allowed time difference.

    Returns:
        The nearest keyframe time, or the original target_time if no suitable keyframe is found.
    """
    if not keyframe_times:
        return target_time

    nearest_time = min(keyframe_times, key=lambda x: abs(x - target_time))
    if abs(nearest_time - target_time) <= tolerance:
        return nearest_time
    return target_time
