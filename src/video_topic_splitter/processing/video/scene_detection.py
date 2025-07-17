#!/usr/bin/env python3
"""PySceneDetect integration for scene and frame extraction."""

import logging
import os
import subprocess
from typing import Dict, List

import imagehash
from PIL import Image

logger = logging.getLogger(__name__)


def extract_unique_frames_from_scenes(
    video_path: str,
    output_dir: str,
    num_frames_per_scene: int = 1,
    hash_threshold: int = 5,
) -> List[Dict]:
    """
    Detects scenes in a video and extracts a specified number of unique frames
    from each scene using PySceneDetect and ImageHash.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory to save the extracted frames.
        num_frames_per_scene: The number of frames to extract from each scene.
        hash_threshold: The threshold for considering two images as duplicates.
                        Lower values mean stricter similarity.

    Returns:
        A list of dictionaries, where each dictionary represents a scene and
        contains the scene number and paths to the unique extracted frames.
    """
    os.makedirs(output_dir, exist_ok=True)
    scene_info = []
    frame_hashes = set()

    # Step 1: Detect scenes and get the list in CSV format
    scenes_csv_path = os.path.join(output_dir, "scenes.csv")
    detect_command = [
        "scenedetect",
        "--input",
        video_path,
        "--output",
        output_dir,
        "detect-content",
        "list-scenes",
        "--output",
        scenes_csv_path,
        "--quiet",
    ]
    try:
        subprocess.run(detect_command, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"Error running scenedetect for scene detection: {e}")
        return []

    # Step 2: Extract frames from each detected scene
    extract_command = [
        "scenedetect",
        "--input",
        video_path,
        "--output",
        output_dir,
        "save-images",
        "--num-images",
        str(num_frames_per_scene),
        "--output",
        os.path.join(output_dir, "frames"), # Save frames in a sub-folder
    ]
    try:
        subprocess.run(extract_command, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"Error running scenedetect for frame extraction: {e}")
        return []

    # Step 3: Filter for unique frames using image hashing
    with open(scenes_csv_path, "r") as f:
        # Skip header lines
        for _ in range(2):
            next(f)
        for line in f:
            parts = line.strip().split(",")
            scene_num = int(parts[0])
            
            scene_data = {"scene_id": scene_num, "frame_paths": []}
            
            # Construct expected frame filenames
            for i in range(1, num_frames_per_scene + 1):
                frame_filename = f"frames/Scene-{scene_num:03d}-{i:02d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)

                if os.path.exists(frame_path):
                    try:
                        with Image.open(frame_path) as img:
                            h = imagehash.phash(img)
                            
                        # Check for hash similarity
                        is_duplicate = False
                        for existing_hash in frame_hashes:
                            if abs(h - existing_hash) <= hash_threshold:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            scene_data["frame_paths"].append(frame_path)
                            frame_hashes.add(h)
                        else:
                            logger.info(f"Skipping duplicate frame: {frame_path}")
                            # Optionally, delete the duplicate frame file
                            os.remove(frame_path)

                    except Exception as e:
                        logger.error(f"Error processing frame {frame_path}: {e}")
            
            if scene_data["frame_paths"]:
                scene_info.append(scene_data)

    return scene_info