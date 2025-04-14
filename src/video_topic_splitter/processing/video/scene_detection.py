#!/usr/bin/env python3
"""Scene detection functionality using the PySceneDetect library."""

import csv
import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2 # Imported but seems unused directly
import numpy as np # Imported but seems unused directly
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import save_images
from scenedetect.stats_manager import StatsManager
from scenedetect.video_splitter import split_video_ffmpeg # Imported but seems unused

logger = logging.getLogger(__name__)


def detect_scenes(
    video_path: str,
    output_dir: str,
    min_scene_len: float = 1.0,
    threshold: int = 27,
    scene_list_path: Optional[str] = None,
) -> List[Tuple[float, float]]:
    """
    Detects scenes in a video using PySceneDetect's content-based detector
    or loads scenes from a pre-existing CSV file.

    If `scene_list_path` is provided and valid, it reads scene start/end times
    from the CSV. Otherwise, it uses `scenedetect.detectors.ContentDetector`
    to find scene changes based on shifts in content. Detected scenes (or loaded
    scenes) are saved to a 'scenes.csv' file in the `output_dir`.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where the 'scenes.csv' file will be saved.
                          The directory will be created if it doesn't exist.
        min_scene_len (float, optional): Minimum duration (in seconds) for a
                                         detected scene. Shorter segments will be
                                         merged with adjacent scenes. Defaults to 1.0.
        threshold (int, optional): Detection threshold for the ContentDetector.
                                   Lower values detect more scene changes (more sensitive).
                                   Higher values detect fewer changes (less sensitive).
                                   Defaults to 27.
        scene_list_path (Optional[str], optional): Path to a CSV file containing
                                                   pre-defined scene boundaries.
                                                   Expected format: Header row, then rows with
                                                   Start Time (sec) in column 4 (index 3) and
                                                   End Time (sec) in column 5 (index 4).
                                                   If provided and valid, scene detection is skipped.
                                                   Defaults to None.

    Returns:
        List[Tuple[float, float]]: A list of tuples, where each tuple represents a scene
                                   and contains the start time and end time in seconds.
                                   Example: [(0.0, 15.5), (15.5, 45.2), ...]

    Raises:
        FileNotFoundError: If `video_path` does not exist.
        RuntimeError: If scene detection fails due to an internal PySceneDetect error
                      or if reading the `scene_list_path` fails unexpectedly.
        csv.Error: If `scene_list_path` is provided but cannot be parsed as CSV.
    """
    scene_boundaries_sec: List[Tuple[float, float]] = []
    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

    # Attempt to load from existing CSV first
    if scene_list_path:
        try:
            with open(scene_list_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)  # Skip header row
                logger.info(f"Attempting to load scenes from CSV: {scene_list_path}")
                # Try to find columns by name, otherwise assume index
                try:
                    start_col = header.index("Start Time (sec)")
                    end_col = header.index("End Time (sec)")
                except ValueError:
                    logger.warning("CSV header 'Start Time (sec)' or 'End Time (sec)' not found, assuming columns 4 and 5.")
                    start_col, end_col = 3, 4 # 0-based index

                for i, row in enumerate(reader):
                    try:
                        if len(row) > max(start_col, end_col):
                            start_time = float(row[start_col])
                            end_time = float(row[end_col])
                            scene_boundaries_sec.append((start_time, end_time))
                        else:
                             logger.warning(f"Skipping row {i+1} in scene list: Not enough columns ({len(row)}). Row: {row}")
                    except (ValueError, IndexError) as e:
                        logger.warning(
                            f"Skipping invalid row {i+1} in scene list: {row} - {str(e)}"
                        )

            if scene_boundaries_sec:
                logger.info(
                    "Loaded %d scenes from %s", len(scene_boundaries_sec), scene_list_path
                )
                # Save the loaded scenes to the standard output CSV as well
                csv_path = os.path.join(output_dir, "scenes.csv")
                try:
                    video_for_fps = open_video(video_path)
                    fps = video_for_fps.frame_rate
                    video_for_fps.release() # Close video handle
                    with open(csv_path, "w", newline="", encoding="utf-8") as f_out:
                        writer = csv.writer(f_out)
                        writer.writerow([
                            "Scene", "Start Frame", "End Frame",
                            "Start Time (sec)", "End Time (sec)", "Duration (sec)"
                        ])
                        for i, (start_sec, end_sec) in enumerate(scene_boundaries_sec):
                            start_frame = int(start_sec * fps)
                            end_frame = int(end_sec * fps)
                            duration = end_sec - start_sec
                            writer.writerow([
                                i + 1, start_frame, end_frame,
                                start_sec, end_sec, duration
                            ])
                    logger.info("Saved loaded scene list to %s", csv_path)
                except Exception as e_save:
                     logger.error(f"Error saving loaded scene list to CSV: {e_save}")

                return scene_boundaries_sec
            else:
                logger.warning(f"Scene list file provided ({scene_list_path}) but no valid scenes found. Proceeding with detection.")

        except FileNotFoundError:
            logger.warning("Scene list file not found: %s. Proceeding with detection.", scene_list_path)
        except (csv.Error, Exception) as e:
            logger.error(
                "Error reading scene list from %s: %s. Proceeding with detection.", scene_list_path, str(e)
            )
            # Fall through to detection if loading fails

    # Perform scene detection if not loaded from CSV
    video = None # Initialize video object
    try:
        # Open video using PySceneDetect
        video = open_video(video_path)
        fps = video.frame_rate
        if not fps or fps <= 0:
             raise ValueError("Invalid or zero framerate detected.")

        # Convert min_scene_len from seconds to frames based on actual video framerate
        min_scene_len_frames = int(min_scene_len * fps)

        # Create scene manager and add the content detector
        stats_manager = StatsManager()
        scene_manager = SceneManager(stats_manager)
        scene_manager.add_detector(
            ContentDetector(threshold=threshold, min_scene_len=min_scene_len_frames)
        )

        # Detect scenes
        logger.info("Detecting scenes in %s (threshold=%d, min_len=%.2f sec / %d frames)...",
                    video_path, threshold, min_scene_len, min_scene_len_frames)
        scene_manager.detect_scenes(video=video, show_progress=True)

        # Get scene list (list of tuples, each tuple is start/end Timecode)
        scene_list_timecodes = scene_manager.get_scene_list() # Returns List[Tuple[Timecode, Timecode]]

        # Convert Timecode objects to seconds
        scene_boundaries_sec = [
            (start.get_seconds(), end.get_seconds())
            for start, end in scene_list_timecodes
        ]

        # Save scene list to CSV
        csv_path = os.path.join(output_dir, "scenes.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Scene", "Start Frame", "End Frame",
                "Start Time (sec)", "End Time (sec)", "Duration (sec)"
            ])
            for i, (start_time, end_time) in enumerate(scene_boundaries_sec):
                # Get corresponding frame numbers from Timecode objects
                start_frame = scene_list_timecodes[i][0].get_frames()
                end_frame = scene_list_timecodes[i][1].get_frames()
                duration = end_time - start_time
                writer.writerow([
                    i + 1, start_frame, end_frame,
                    start_time, end_time, duration
                ])

        logger.info("Detected %d scenes.", len(scene_boundaries_sec))
        logger.info("Scene list saved to %s", csv_path)

        return scene_boundaries_sec

    except Exception as e:
        logger.error("Error during scene detection for %s: %s", video_path, str(e), exc_info=True)
        # Wrap specific PySceneDetect errors or raise a general one
        raise RuntimeError(f"Scene detection failed for {video_path}: {str(e)}") from e
    finally:
        # Ensure video object is released if it was opened
        if video:
            video.release()


def extract_scene_frames(
    video_path: str,
    scene_boundaries: List[Tuple[float, float]],
    output_dir: str,
    num_frames_per_scene: int = 1,
    jpg_quality: int = 95,
) -> List[Dict]:
    """
    Extracts representative frames (thumbnails) from each detected scene.

    Uses PySceneDetect's `save_images` function to efficiently extract frames
    spread throughout each scene duration.

    Args:
        video_path (str): Path to the input video file.
        scene_boundaries (List[Tuple[float, float]]): A list of scene boundaries
                                                      (start_time_sec, end_time_sec)
                                                      as returned by `detect_scenes`.
        output_dir (str): Directory where the extracted frame images will be saved.
                          The directory will be created if it doesn't exist.
        num_frames_per_scene (int, optional): The number of frames to extract
                                              from each scene. Frames will be
                                              distributed evenly within the scene.
                                              Defaults to 1.
        jpg_quality (int, optional): The quality setting (1-100) for saved JPEG images.
                                     Defaults to 95.

    Returns:
        List[Dict]: A list of dictionaries, one for each scene. Each dictionary
                    contains:
                    - 'scene_id' (int): The 1-based index of the scene.
                    - 'start_time' (float): Start time of the scene in seconds.
                    - 'end_time' (float): End time of the scene in seconds.
                    - 'duration' (float): Duration of the scene in seconds.
                    - 'frame_paths' (List[str]): A list of absolute file paths to the
                                                 extracted frames for this scene.

    Raises:
        FileNotFoundError: If `video_path` does not exist.
        RuntimeError: If frame extraction fails due to an internal PySceneDetect
                      error or file system issues.
        ValueError: If `scene_boundaries` is empty.
    """
    if not scene_boundaries:
        logger.warning("No scene boundaries provided for frame extraction.")
        return []

    video = None # Initialize video object
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Open video using PySceneDetect
        video = open_video(video_path)
        fps = video.frame_rate
        if not fps or fps <= 0:
             raise ValueError("Invalid or zero framerate detected for frame extraction.")

        # Convert scene boundaries from seconds back to PySceneDetect Timecode objects
        # This is needed for save_images function
        scene_list_timecodes = []
        for start_sec, end_sec in scene_boundaries:
            start_tc = video.base_timecode + int(start_sec * fps)
            end_tc = video.base_timecode + int(end_sec * fps)
            # Ensure end frame is at least start frame + 1 for valid duration
            if end_tc.get_frames() <= start_tc.get_frames():
                 logger.warning(f"Scene from {start_sec:.2f}s to {end_sec:.2f}s has zero or negative duration in frames. Adjusting end frame.")
                 end_tc = start_tc + 1 # Make it at least one frame long
            scene_list_timecodes.append((start_tc, end_tc))


        logger.info("Extracting %d frame(s) per scene from %d scenes...",
                    num_frames_per_scene, len(scene_list_timecodes))

        # Use PySceneDetect's save_images function
        # It returns a dictionary mapping scene index (0-based) to list of frame filenames
        # Note: PySceneDetect >= v0.6 uses 'video' directly, older versions might need 'video.capture'
        # The API for save_images also changed. Let's use keyword arguments for clarity.
        image_filenames_dict = save_images(
            scene_list=scene_list_timecodes,
            video=video, # Pass the video backend object
            output_dir=output_dir,
            num_images=num_frames_per_scene,
            image_extension="jpg",
            encoder_options={'quality': jpg_quality}, # Pass quality via encoder_options
            file_name_template='$SCENE_NUMBER-$IMAGE_NUMBER', # Use $SCENE_NUMBER for 1-based index
            show_progress=True,
        )

        # Create scene information list with absolute frame paths
        scene_info = []
        total_frames_extracted = 0
        for i, (start_time, end_time) in enumerate(scene_boundaries):
            scene_idx = i # 0-based index used by save_images dict keys
            # Get the list of filenames for this scene (relative to output_dir)
            relative_frame_paths = image_filenames_dict.get(scene_idx, [])
            # Convert to absolute paths
            absolute_frame_paths = [os.path.join(output_dir, f) for f in relative_frame_paths]
            total_frames_extracted += len(absolute_frame_paths)

            scene_info.append(
                {
                    "scene_id": i + 1, # 1-based scene ID for reporting
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                    "frame_paths": absolute_frame_paths,
                }
            )

        logger.info(
            "Extracted %d total frames from %d scenes into %s",
            total_frames_extracted,
            len(scene_boundaries),
            output_dir
        )

        return scene_info

    except Exception as e:
        logger.error("Error extracting scene frames from %s: %s", video_path, str(e), exc_info=True)
        raise RuntimeError(f"Frame extraction failed for {video_path}: {str(e)}") from e
    finally:
         # Ensure video object is released if it was opened
        if video:
            video.release()


def extract_scenes_from_video(
    video_path: str,
    output_dir: str,
    min_scene_len: float = 1.0,
    threshold: int = 27,
    num_frames_per_scene: int = 1,
    jpg_quality: int = 95,
    scene_list_path: Optional[str] = None, # Added parameter
) -> List[Dict]:
    """
    Orchestrates the full scene analysis process: detects scenes and extracts
    representative frames.

    Combines the functionality of `detect_scenes` and `extract_scene_frames`.
    It first detects (or loads) scene boundaries and then extracts a specified
    number of frames from each detected scene.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Base directory where results will be saved. A 'scenes'
                          subdirectory will be created within this directory to store
                          the 'scenes.csv' file and the extracted frame images.
        min_scene_len (float, optional): Minimum scene length in seconds for detection.
                                         Defaults to 1.0.
        threshold (int, optional): ContentDetector threshold for scene detection.
                                   Defaults to 27.
        num_frames_per_scene (int, optional): Number of frames to extract per scene.
                                              Defaults to 1.
        jpg_quality (int, optional): JPEG quality for extracted frames (1-100).
                                     Defaults to 95.
        scene_list_path (Optional[str], optional): Path to a pre-existing CSV file
                                                   with scene boundaries to load instead
                                                   of performing detection. Defaults to None.

    Returns:
        List[Dict]: A list of dictionaries, one for each scene, containing scene
                    information (ID, start, end, duration) and the paths to the
                    extracted frames for that scene. Returns an empty list if no
                    scenes are detected or if an error occurs.

    Raises:
        RuntimeError: If either scene detection or frame extraction fails.
        FileNotFoundError: If the input `video_path` is not found.
    """
    # Create a dedicated subdirectory for scene-related outputs
    scenes_output_dir = os.path.join(output_dir, "scenes")
    os.makedirs(scenes_output_dir, exist_ok=True)
    logger.info(f"Scene analysis outputs will be saved in: {scenes_output_dir}")

    try:
        # 1. Detect (or load) scenes
        scene_boundaries = detect_scenes(
            video_path=video_path,
            output_dir=scenes_output_dir, # Save scenes.csv here
            min_scene_len=min_scene_len,
            threshold=threshold,
            scene_list_path=scene_list_path # Pass through the optional path
        )

        if not scene_boundaries:
            logger.warning("No scenes were detected or loaded. Cannot extract frames.")
            return []

        # 2. Extract frames based on detected boundaries
        scene_info = extract_scene_frames(
            video_path=video_path,
            scene_boundaries=scene_boundaries,
            output_dir=scenes_output_dir, # Save frames here too
            num_frames_per_scene=num_frames_per_scene,
            jpg_quality=jpg_quality
        )

        return scene_info

    except (RuntimeError, FileNotFoundError) as e:
        logger.error(f"Failed to extract scenes from video {video_path}: {e}")
        # Re-raise the exception to signal failure
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during scene extraction from {video_path}: {e}", exc_info=True)
        # Raise a runtime error for unexpected issues
        raise RuntimeError(f"Unexpected error during scene extraction: {e}") from e

