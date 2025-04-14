#!/usr/bin/env python3
"""Scene detection and video splitting functionality using PySceneDetect."""

import csv
import logging
import os
from typing import Dict, List, Optional, Tuple

from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector, AdaptiveDetector
from scenedetect.scene_manager import save_images # Keep for frame extraction
from scenedetect.stats_manager import StatsManager
from scenedetect.video_splitter import split_video_ffmpeg # Import for splitting

logger = logging.getLogger(__name__)


def detect_scenes(
    video_path: str,
    output_dir: str, # Base output dir for scene-related files (CSV, frames)
    threshold: float = 27.0,
    min_scene_len_sec: float = 1.0,
    save_csv: bool = True,
    short_video_threshold_sec: float = 60.0,  # Videos <= this length are considered "short"
) -> List[Tuple[float, float]]:
    """
    Detects scenes in a video using PySceneDetect's content-based detector.
    If no scenes are detected, falls back to AdaptiveDetector.
    For short videos with no detected scenes, creates a single scene spanning the entire video.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where scene-related files (like scenes.csv)
                          will be saved. Created if it doesn't exist.
        threshold (float, optional): Detection threshold for the ContentDetector.
                                     Lower values detect more changes. Defaults to 27.0.
        min_scene_len_sec (float, optional): Minimum duration (in seconds) for a
                                             detected scene. Defaults to 1.0.
        save_csv (bool, optional): Whether to save the detected scene list to a
                                   CSV file in the output_dir. Defaults to True.
        short_video_threshold_sec (float, optional): Maximum duration in seconds for a video
                                                    to be considered "short". Defaults to 60.0.

    Returns:
        List[Tuple[float, float]]: A list of tuples, where each tuple represents a scene
                                   and contains the start time and end time in seconds.
                                   Example: [(0.0, 15.5), (15.5, 45.2), ...]

    Raises:
        FileNotFoundError: If `video_path` does not exist.
        RuntimeError: If scene detection fails due to an internal PySceneDetect error.
    """
    scene_boundaries_sec: List[Tuple[float, float]] = []
    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

    video = None # Initialize video object
    try:
        # Open video using PySceneDetect
        video = open_video(video_path)
        fps = video.frame_rate
        if not fps or fps <= 0:
             raise ValueError("Invalid or zero framerate detected.")
             
        # Get video duration in seconds
        video_duration_sec = video.duration.get_seconds()
        logger.info(f"Video duration: {video_duration_sec:.2f} seconds")

        # Convert min_scene_len from seconds to frames
        min_scene_len_frames = int(min_scene_len_sec * fps)

        # Create scene manager and add the content detector
        stats_manager = StatsManager()
        scene_manager = SceneManager(stats_manager)
        scene_manager.add_detector(
            ContentDetector(threshold=threshold, min_scene_len=min_scene_len_frames)
        )

        # Detect scenes
        logger.info("Detecting scenes in %s (threshold=%.1f, min_len=%.2f sec)...",
                    video_path, threshold, min_scene_len_sec)
        # Use default show_progress=False for cleaner logs unless debugging
        scene_manager.detect_scenes(video=video, show_progress=False)

        # Get scene list (list of tuples, each tuple is start/end Timecode)
        scene_list_timecodes = scene_manager.get_scene_list()

        # If no scenes detected with ContentDetector, try AdaptiveDetector
        if not scene_list_timecodes:
            logger.info("No scenes detected with ContentDetector. Trying AdaptiveDetector...")
            
            # Reset scene manager with a new one
            scene_manager = SceneManager(StatsManager())
            
            # Add AdaptiveDetector with appropriate parameters
            # Lower adaptive_threshold makes it more sensitive
            scene_manager.add_detector(
                AdaptiveDetector(
                    adaptive_threshold=3.0,  # Default is 3.0
                    min_scene_len=min_scene_len_frames,
                    window_width=2,  # Default is 2
                    min_content_val=15.0  # Default is 15.0
                )
            )
            
            # Detect scenes again with AdaptiveDetector
            logger.info("Detecting scenes with AdaptiveDetector...")
            scene_manager.detect_scenes(video=video, show_progress=False)
            scene_list_timecodes = scene_manager.get_scene_list()
            
            if scene_list_timecodes:
                logger.info("AdaptiveDetector found %d scenes.", len(scene_list_timecodes))
            else:
                logger.warning("No scenes detected with AdaptiveDetector either.")
                
                # For short videos, create a single scene spanning the entire video
                if video_duration_sec <= short_video_threshold_sec:
                    logger.info(f"Short video detected ({video_duration_sec:.2f} sec). Creating a single scene for the entire video.")
                    start_tc = video.base_timecode
                    end_tc = video.base_timecode + int(video_duration_sec * fps)
                    scene_list_timecodes = [(start_tc, end_tc)]
                    logger.info("Created 1 scene spanning the entire video.")

        # Convert Timecode objects to seconds
        scene_boundaries_sec = [
            (start.get_seconds(), end.get_seconds())
            for start, end in scene_list_timecodes
        ]

        logger.info("Detected %d scenes.", len(scene_boundaries_sec))

        # Save scene list to CSV if requested
        if save_csv and scene_boundaries_sec:
            csv_path = os.path.join(output_dir, "scenes.csv")
            try:
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "Scene", "Start Frame", "End Frame",
                        "Start Time (sec)", "End Time (sec)", "Duration (sec)"
                    ])
                    for i, (start_time, end_time) in enumerate(scene_boundaries_sec):
                        start_frame = scene_list_timecodes[i][0].get_frames()
                        end_frame = scene_list_timecodes[i][1].get_frames()
                        duration = end_time - start_time
                        writer.writerow([
                            i + 1, start_frame, end_frame,
                            start_time, end_time, duration
                        ])
                logger.info("Scene list saved to %s", csv_path)
            except Exception as e_save:
                 logger.error(f"Error saving scene list to CSV: {e_save}")

        return scene_boundaries_sec

    except Exception as e:
        logger.error("Error during scene detection for %s: %s", video_path, str(e), exc_info=True)
        raise RuntimeError(f"Scene detection failed for {video_path}: {str(e)}") from e
    finally:
        # Safely close video if it has a release method
        if video and hasattr(video, 'release'):
            video.release()


def extract_scene_frames(
    video_path: str,
    scene_boundaries: List[Tuple[float, float]],
    output_dir: str, # Dir where frame images will be saved
    num_frames_per_scene: int = 1,
    frame_format: str = "jpg",
    jpg_quality: int = 90,
    short_video_frames: int = 3,  # Number of frames to extract for short videos
) -> List[Dict]:
    """
    Extracts representative frames (thumbnails) from each detected scene.
    For short videos with a single scene, extracts multiple frames.

    Args:
        video_path (str): Path to the input video file.
        scene_boundaries (List[Tuple[float, float]]): A list of scene boundaries
                                                      (start_time_sec, end_time_sec).
        output_dir (str): Directory where the extracted frame images will be saved.
                          Created if it doesn't exist.
        num_frames_per_scene (int, optional): Number of frames to extract per scene.
                                              Frames are distributed evenly. Defaults to 1.
        frame_format (str, optional): Format for saved frames ('jpg' or 'png').
                                      Defaults to "jpg".
        jpg_quality (int, optional): Quality for saved JPEG images (1-100).
                                     Defaults to 90.
        short_video_frames (int, optional): Number of frames to extract for short videos
                                           with a single scene. Defaults to 3.

    Returns:
        List[Dict]: A list of dictionaries, one for each scene, containing:
                    - 'scene_id' (int): 1-based index.
                    - 'start_time' (float): Scene start time (sec).
                    - 'end_time' (float): Scene end time (sec).
                    - 'duration' (float): Scene duration (sec).
                    - 'frame_paths' (List[str]): Absolute paths to extracted frames.

    Raises:
        FileNotFoundError: If `video_path` does not exist.
        RuntimeError: If frame extraction fails.
        ValueError: If `scene_boundaries` is empty or video has invalid framerate.
    """
    if not scene_boundaries:
        logger.warning("No scene boundaries provided for frame extraction.")
        return []

    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists
    video = None
    try:
        video = open_video(video_path)
        fps = video.frame_rate
        if not fps or fps <= 0:
             raise ValueError("Invalid or zero framerate detected for frame extraction.")

        # Check if this is a single scene spanning a short video
        is_short_single_scene = False
        if len(scene_boundaries) == 1:
            start_time, end_time = scene_boundaries[0]
            duration = end_time - start_time
            if duration <= 60.0:  # 60 seconds threshold for short videos
                is_short_single_scene = True
                logger.info(f"Detected short video with single scene ({duration:.2f} sec). Will extract {short_video_frames} frames.")
                num_frames_to_extract = short_video_frames
            else:
                num_frames_to_extract = num_frames_per_scene
        else:
            num_frames_to_extract = num_frames_per_scene

        # Convert scene boundaries from seconds back to PySceneDetect Timecode objects
        scene_list_timecodes = []
        for start_sec, end_sec in scene_boundaries:
            start_tc = video.base_timecode + int(start_sec * fps)
            end_tc = video.base_timecode + int(end_sec * fps)
            # Ensure end frame is valid
            if end_tc.get_frames() <= start_tc.get_frames():
                 end_tc = start_tc + 1 # Min 1 frame duration
            scene_list_timecodes.append((start_tc, end_tc))

        logger.info("Extracting %d frame(s) per scene from %d scenes into %s...",
                    num_frames_to_extract, len(scene_list_timecodes), output_dir)

        # Use PySceneDetect's save_images
        image_filenames_dict = save_images(
            scene_list=scene_list_timecodes,
            video=video,
            output_dir=output_dir,
            num_images=num_frames_to_extract,
            image_extension=frame_format,
            image_name_template='$SCENE_NUMBER-$IMAGE_NUMBER', # 1-based scene number
            show_progress=False, # Less verbose logs
        )

        # Create scene information list with absolute frame paths
        scene_info = []
        total_frames_extracted = 0
        for i, (start_time, end_time) in enumerate(scene_boundaries):
            scene_idx = i # 0-based index used by save_images dict keys
            relative_frame_paths = image_filenames_dict.get(scene_idx, [])
            absolute_frame_paths = [os.path.join(output_dir, f) for f in relative_frame_paths]
            total_frames_extracted += len(absolute_frame_paths)

            scene_info.append(
                {
                    "scene_id": i + 1, # 1-based scene ID
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                    "frame_paths": absolute_frame_paths,
                }
            )

        logger.info(
            "Extracted %d total frames from %d scenes.",
            total_frames_extracted, len(scene_boundaries)
        )
        return scene_info

    except Exception as e:
        logger.error("Error extracting scene frames from %s: %s", video_path, str(e), exc_info=True)
        raise RuntimeError(f"Frame extraction failed for {video_path}: {str(e)}") from e
    finally:
        # Safely close video if it has a release method
        if video and hasattr(video, 'release'):
            video.release()


def split_video_by_scenes(
    video_path: str,
    scene_list: List[Tuple[float, float]],
    output_dir: str,
    output_file_template: str = 'scene_$SCENE_NUMBER.mp4',
    show_progress: bool = False,
    suppress_output: bool = True, # Keep ffmpeg logs quieter by default
) -> List[str]:
    """
    Splits a video into multiple files based on a list of detected scenes.

    Uses PySceneDetect's integration with FFmpeg (`split_video_ffmpeg`).

    Args:
        video_path (str): Path to the input video file.
        scene_list (List[Tuple[float, float]]): List of scene boundaries
                                                (start_sec, end_sec).
        output_dir (str): Directory where the split video files will be saved.
                          Created if it doesn't exist.
        output_file_template (str, optional): Template for naming output files.
                                              Use $SCENE_NUMBER for 1-based index.
                                              Defaults to 'scene_$SCENE_NUMBER.mp4'.
        show_progress (bool, optional): Show FFmpeg progress. Defaults to False.
        suppress_output (bool, optional): Suppress FFmpeg non-progress output.
                                          Defaults to True.

    Returns:
        List[str]: A list of file paths for the created video segments.

    Raises:
        FileNotFoundError: If `video_path` does not exist.
        RuntimeError: If video splitting fails (e.g., FFmpeg error).
    """
    if not scene_list:
        logger.warning("No scenes provided for splitting.")
        return []

    # For a single scene that spans the entire video, just return the original video
    if len(scene_list) == 1:
        start_time, end_time = scene_list[0]
        video = open_video(video_path)
        video_duration = video.duration.get_seconds()
        
        # If the scene covers almost the entire video (within 0.5 seconds tolerance)
        if abs(start_time) < 0.5 and abs(end_time - video_duration) < 0.5:
            logger.info("Single scene spans the entire video. Skipping unnecessary splitting.")
            return [video_path]

    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists
    video = None
    try:
        video = open_video(video_path)
        fps = video.frame_rate
        if not fps or fps <= 0:
            raise ValueError("Invalid or zero framerate detected for video splitting.")

        # Convert scene boundaries to Timecode objects
        scene_list_timecodes = []
        for start_sec, end_sec in scene_list:
            start_tc = video.base_timecode + int(start_sec * fps)
            end_tc = video.base_timecode + int(end_sec * fps)
            if end_tc.get_frames() <= start_tc.get_frames():
                end_tc = start_tc + 1 # Min 1 frame duration
            scene_list_timecodes.append((start_tc, end_tc))

        logger.info("Splitting video into %d scenes in directory: %s",
                    len(scene_list_timecodes), output_dir)

        # Perform the split using PySceneDetect's helper function
        split_files = split_video_ffmpeg(
            video_path=video_path,
            scene_list=scene_list_timecodes,
            output_file_template=os.path.join(output_dir, output_file_template),
            show_progress=show_progress,
            suppress_output=suppress_output,
            # Add other ffmpeg args if needed via `ffmpeg_args=['-map', '0', '-c', 'copy']`
            # Using default copy codec for speed. Re-encoding might be needed for format changes.
        )

        # Verify files were created (split_video_ffmpeg returns paths even on error sometimes)
        created_files = [f for f in split_files if os.path.exists(f) and os.path.getsize(f) > 0]

        if len(created_files) != len(scene_list_timecodes):
             logger.warning(f"Expected {len(scene_list_timecodes)} split files, but found {len(created_files)}. Check FFmpeg logs.")
             # Potentially raise an error here if exact matching is critical

        logger.info("Video splitting complete. Created %d segment files.", len(created_files))
        return created_files

    except Exception as e:
        logger.error("Error during video splitting: %s", str(e), exc_info=True)
        raise RuntimeError(f"Video splitting failed for {video_path}: {str(e)}") from e
    finally:
        # Safely close video if it has a release method
        if video and hasattr(video, 'release'):
            video.release()

