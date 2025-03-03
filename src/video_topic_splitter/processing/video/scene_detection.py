#!/usr/bin/env python3
"""Scene detection functionality using PySceneDetect."""

import csv
import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import save_images
from scenedetect.stats_manager import StatsManager
from scenedetect.video_splitter import split_video_ffmpeg

logger = logging.getLogger(__name__)


def detect_scenes(
    video_path: str,
    output_dir: str,
    min_scene_len: float = 1.0,
    threshold: int = 27,
    scene_list_path: Optional[str] = None,
) -> List[Tuple[float, float]]:
    """
    Detect scenes in a video using PySceneDetect's content detector.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save scene information
        min_scene_len: Minimum scene length in seconds
        threshold: Threshold for content detector (lower is more sensitive)
        scene_list_path: Optional path to a CSV file containing a pre-existing scene list.

    Returns:
        List of scene boundaries as (start_time, end_time) in seconds
    """
    scene_boundaries: List[Tuple[float, float]] = []
    if scene_list_path:
        try:
            import csv

            with open(scene_list_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header row
                for row in reader:
                    try:
                        start_time = float(row[3])  # Start Time (sec)
                        end_time = float(row[4])  # End Time (sec)
                        scene_boundaries.append((start_time, end_time))
                    except (ValueError, IndexError) as e:
                        logger.warning(
                            f"Skipping invalid row in scene list: {row} - {str(e)}"
                        )
            logger.info(
                "Loaded %s scenes from %s", len(scene_boundaries), scene_list_path
            )
            return scene_boundaries
        except FileNotFoundError:
            logger.warning("Scene list file not found: %s", scene_list_path)
        except (Exception, csv.Error) as e:
            logger.error(
                "Error reading scene list from %s: %s", scene_list_path, str(e)
            )

    try:
        # Open video first to get frame rate
        video = open_video(video_path)

        # Convert min_scene_len from seconds to frames
        min_scene_len_frames = int(min_scene_len * video.frame_rate)

        # Create scene and stats manager
        stats_manager = StatsManager()
        scene_manager = SceneManager(stats_manager)

        # Add content detector with min_scene_len in frames
        scene_manager.add_detector(
            ContentDetector(threshold=threshold, min_scene_len=min_scene_len_frames)
        )

        # Detect scenes
        logger.info("Detecting scenes in %s...", video_path)
        scene_manager.detect_scenes(video, show_progress=True)

        # Get scene list
        scene_list = scene_manager.get_scene_list()

        # Convert frame numbers to timestamps
        fps = video.frame_rate

        logger.info(
            "Scene list type: %s, length: %s", type(scene_list), len(scene_list)
        )
        if scene_list:
            logger.info(
                "First scene type: %s, value: %s", type(scene_list[0]), scene_list[0]
            )

        for scene in scene_list:
            start_frame, end_frame = scene
            logger.info(
                "Frame types: start_frame=%s, end_frame=%s",
                type(start_frame),
                type(end_frame),
            )

        # Save scene list to CSV
        csv_path = os.path.join(output_dir, "scenes.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(
                "Scene,Start Frame,End Frame,Start Time (sec),End Time (sec),Duration (sec)\n"
            )
            for i, (start_time, end_time) in enumerate(scene_boundaries):
                start_frame = int(start_time * fps)
                end_frame = int(end_time * fps)
                duration = end_time - start_time
                f.write(
                    "%s,%s,%s,%s,%s,%s\n"
                    % (i + 1, start_frame, end_frame, start_time, end_time, duration)
                )

        logger.info("Detected %s scenes", len(scene_boundaries))
        logger.info("Scene list saved to %s", csv_path)

        return scene_boundaries

    except Exception as e:
        logger.error("Error detecting scenes: %s", str(e))
        raise RuntimeError("Scene detection failed: %s", str(e)) from e


def extract_scene_frames(
    video_path: str,
    scene_boundaries: List[Tuple[float, float]],
    output_dir: str,
    num_frames_per_scene: int = 1,
    jpg_quality: int = 95,
) -> List[Dict]:
    """
    Extract representative frames from each scene.

    Args:
        video_path: Path to the video file
        scene_boundaries: List of scene boundaries (start_time, end_time)
        output_dir: Directory to save extracted frames
        num_frames_per_scene: Number of frames to extract per scene
        jpg_quality: JPEG quality (1-100)

    Returns:
        List of dictionaries with scene information and frame paths
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Open video
        video = open_video(video_path)
        fps = video.frame_rate

        # Convert scene boundaries to frame numbers
        scene_list = []
        for start_time, end_time in scene_boundaries:
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            scene_list.append((start_frame, end_frame))

        # Extract frames
        logger.info("Extracting %s frame(s) per scene...", num_frames_per_scene)

        # Add debug logging
        logger.info("Video object type: %s", type(video))
        logger.info("Video object attributes: %s", dir(video))

        # From the debug output, we can see the video object has a 'capture' attribute
        # which is likely what save_images expects

        # Check PySceneDetect version to handle API changes
        import scenedetect

        logger.info("PySceneDetect version: %s", scenedetect.__version__)

        try:
            # Use PySceneDetect's save_images function with the correct parameters
            # Based on the error, there seems to be a conflict with the num_images parameter
            image_filenames = save_images(
                video.capture,  # Use the capture attribute instead
                scene_list,
                output_dir,
                num_images=num_frames_per_scene,
                image_extension="jpg",
                image_name_template="scene-$SCENE_NUMBER-$IMAGE_NUMBER",
                quality=jpg_quality,
                show_progress=True,
            )
        except TypeError as e:
            # If that fails, try an alternative approach based on the error message
            logger.info(
                "First save_images attempt failed: %s, trying alternative approach",
                str(e),
            )

            # Try with positional arguments only for the first few parameters
            image_filenames = save_images(
                video.capture,
                scene_list,
                output_dir,
                num_frames_per_scene,  # Positional instead of keyword
                image_extension="jpg",
                image_name_template="scene-$SCENE_NUMBER-$IMAGE_NUMBER",
                quality=jpg_quality,
                show_progress=True,
            )

        # Create scene information with frame paths
        scene_info = []
        for i, ((start_time, end_time), frame_paths) in enumerate(
            zip(scene_boundaries, image_filenames)
        ):
            scene_info.append(
                {
                    "scene_id": i + 1,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                    "frame_paths": frame_paths,
                }
            )

        logger.info(
            "Extracted %s frames from %s scenes",
            sum(len(paths) for paths in image_filenames),
            len(scene_boundaries),
        )

        return scene_info

    except Exception as e:
        logger.error("Error extracting scene frames: %s", str(e))
        raise RuntimeError("Frame extraction failed: %s", str(e)) from e


def extract_scenes_from_video(
    video_path: str,
    output_dir: str,
    min_scene_len: float = 1.0,
    threshold: int = 27,
    num_frames_per_scene: int = 1,
    jpg_quality: int = 95,
) -> List[Dict]:
    """
    Detect scenes and extract representative frames from a video.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save scene information and frames
        min_scene_len: Minimum scene length in seconds
        threshold: Threshold for content detector (lower is more sensitive)
        num_frames_per_scene: Number of frames to extract per scene
        jpg_quality: JPEG quality (1-100)

    Returns:
        List of dictionaries with scene information and frame paths
    """
    # Create scenes directory
    scenes_dir = os.path.join(output_dir, "scenes")
    os.makedirs(scenes_dir, exist_ok=True)

    # Detect scenes
    scene_boundaries = detect_scenes(video_path, scenes_dir, min_scene_len, threshold)

    # Extract frames
    scene_info = extract_scene_frames(
        video_path, scene_boundaries, scenes_dir, num_frames_per_scene, jpg_quality
    )

    return scene_info
