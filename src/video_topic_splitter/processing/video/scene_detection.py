#!/usr/bin/env python3
"""Scene detection functionality using PySceneDetect."""

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
) -> List[Tuple[float, float]]:
    """
    Detect scenes in a video using PySceneDetect's content detector.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save scene information
        min_scene_len: Minimum scene length in seconds
        threshold: Threshold for content detector (lower is more sensitive)

    Returns:
        List of scene boundaries as (start_time, end_time) in seconds
    """
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
        logger.info(f"Detecting scenes in {video_path}...")
        scene_manager.detect_scenes(video, show_progress=True)

        # Get scene list
        scene_list = scene_manager.get_scene_list()

        # Convert frame numbers to timestamps
        fps = video.frame_rate
        scene_boundaries = []

        logger.info(f"Scene list type: {type(scene_list)}, length: {len(scene_list)}")
        if scene_list:
            logger.info(
                f"First scene type: {type(scene_list[0])}, value: {scene_list[0]}"
            )

        for scene in scene_list:
            start_frame, end_frame = scene
            logger.info(
                f"Frame types: start_frame={type(start_frame)}, end_frame={type(end_frame)}"
            )

            # Convert FrameTimecode objects to float seconds
            start_time = start_frame.get_seconds()
            end_time = end_frame.get_seconds()
            scene_boundaries.append((start_time, end_time))

        # Save scene list to CSV
        csv_path = os.path.join(output_dir, "scenes.csv")
        with open(csv_path, "w") as f:
            f.write(
                "Scene,Start Frame,End Frame,Start Time (sec),End Time (sec),Duration (sec)\n"
            )
            for i, (start_time, end_time) in enumerate(scene_boundaries):
                start_frame = int(start_time * fps)
                end_frame = int(end_time * fps)
                duration = end_time - start_time
                f.write(
                    f"{i+1},{start_frame},{end_frame},{start_time:.3f},{end_time:.3f},{duration:.3f}\n"
                )

        logger.info(f"Detected {len(scene_boundaries)} scenes")
        logger.info(f"Scene list saved to {csv_path}")

        return scene_boundaries

    except Exception as e:
        logger.error(f"Error detecting scenes: {str(e)}")
        raise RuntimeError(f"Scene detection failed: {str(e)}")


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
        logger.info(f"Extracting {num_frames_per_scene} frame(s) per scene...")

        # Use PySceneDetect's save_images function
        image_filenames = save_images(
            video.cap,
            scene_list,
            output_dir,
            num_images=num_frames_per_scene,
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
            f"Extracted {sum(len(paths) for paths in image_filenames)} frames from {len(scene_boundaries)} scenes"
        )

        return scene_info

    except Exception as e:
        logger.error(f"Error extracting scene frames: {str(e)}")
        raise RuntimeError(f"Frame extraction failed: {str(e)}")


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
