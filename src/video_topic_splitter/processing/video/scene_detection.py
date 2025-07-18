#!/usr/bin/env python3
"""PySceneDetect integration for scene and frame extraction."""

import logging
import os
import subprocess
from typing import Dict, List

import imagehash
from PIL import Image

logger = logging.getLogger(__name__)


def detect_scenes_enhanced(
    video_path: str,
    output_dir: str,
    threshold: float = 27.0,
    min_scene_len_sec: float = 1.0,
    save_csv: bool = True,
    short_video_threshold_sec: float = 60.0,
) -> List[Tuple[float, float]]:
    """
    Enhanced scene detection using dual detector approach with fallback.
    
    Uses ContentDetector primarily, with AdaptiveDetector as fallback.
    Handles short videos intelligently by creating single scene for videos ≤60s.
    
    Args:
        video_path: Path to the input video file
        output_dir: Directory to save scene-related output files
        threshold: Detection threshold for ContentDetector
        min_scene_len_sec: Minimum duration for a scene in seconds
        save_csv: Whether to save detailed scene CSV
        short_video_threshold_sec: Max duration for short video handling
        
    Returns:
        List of (start_time, end_time) tuples in seconds
    """
    try:
        # Try to use PySceneDetect Python API if available
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import ContentDetector, AdaptiveDetector
        from scenedetect.stats_manager import StatsManager
        
        return _detect_scenes_python_api(
            video_path, output_dir, threshold, min_scene_len_sec, 
            save_csv, short_video_threshold_sec
        )
        
    except ImportError:
        logger.warning("PySceneDetect Python API not available, falling back to CLI method")
        return _detect_scenes_cli_fallback(video_path, output_dir)


def _detect_scenes_python_api(
    video_path: str,
    output_dir: str,
    threshold: float,
    min_scene_len_sec: float,
    save_csv: bool,
    short_video_threshold_sec: float
) -> List[Tuple[float, float]]:
    """Enhanced scene detection using PySceneDetect Python API."""
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import ContentDetector, AdaptiveDetector
    from scenedetect.stats_manager import StatsManager
    import csv
    
    scene_boundaries_sec = []
    os.makedirs(output_dir, exist_ok=True)
    
    video = None
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

        # --- Attempt 1: ContentDetector ---
        stats_manager_content = StatsManager()
        scene_manager_content = SceneManager(stats_manager_content)
        scene_manager_content.add_detector(
            ContentDetector(threshold=threshold, min_scene_len=min_scene_len_frames)
        )

        logger.info(
            "Detecting scenes using ContentDetector (threshold=%.1f, min_len=%.2f sec)...",
            threshold, min_scene_len_sec
        )
        scene_manager_content.detect_scenes(video=video, show_progress=False)
        scene_list_timecodes = scene_manager_content.get_scene_list()

        # --- Attempt 2: AdaptiveDetector (Fallback) ---
        if not scene_list_timecodes:
            logger.info("No scenes detected with ContentDetector. Trying AdaptiveDetector...")
            video.reset()  # Reset video position for the next detector

            stats_manager_adaptive = StatsManager()
            scene_manager_adaptive = SceneManager(stats_manager_adaptive)
            scene_manager_adaptive.add_detector(
                AdaptiveDetector(
                    adaptive_threshold=3.0,
                    min_scene_len=min_scene_len_frames,
                )
            )

            scene_manager_adaptive.detect_scenes(video=video, show_progress=False)
            scene_list_timecodes = scene_manager_adaptive.get_scene_list()

            if scene_list_timecodes:
                logger.info("AdaptiveDetector found %d scenes.", len(scene_list_timecodes))
            else:
                logger.warning("No scenes detected with AdaptiveDetector either.")
                
                # --- Handle Short Videos with No Detected Scenes ---
                if video_duration_sec <= short_video_threshold_sec:
                    logger.info(
                        f"Short video detected ({video_duration_sec:.2f} sec <= "
                        f"{short_video_threshold_sec:.1f} sec). Creating a single scene."
                    )
                    start_tc = video.base_timecode
                    end_tc = video.base_timecode + int(video_duration_sec * fps)
                    if end_tc.get_frames() <= start_tc.get_frames():
                        end_tc = start_tc + 1
                    scene_list_timecodes = [(start_tc, end_tc)]
                    logger.info("Created 1 scene spanning the entire video.")

        # Convert Timecode objects to seconds
        scene_boundaries_sec = [
            (start.get_seconds(), end.get_seconds())
            for start, end in scene_list_timecodes
        ]

        logger.info("Detected %d scenes.", len(scene_boundaries_sec))

        # Save CSV output if requested
        if save_csv and scene_boundaries_sec:
            _save_scenes_csv(output_dir, scene_list_timecodes, scene_boundaries_sec)

        return scene_boundaries_sec

    except Exception as e:
        logger.error(f"Error during enhanced scene detection: {e}")
        raise
    finally:
        if video and hasattr(video, 'release'):
            try:
                video.release()
            except Exception as e_release:
                logger.warning(f"Error releasing video handle: {e_release}")


def _detect_scenes_cli_fallback(video_path: str, output_dir: str) -> List[Tuple[float, float]]:
    """Fallback to CLI-based scene detection."""
    os.makedirs(output_dir, exist_ok=True)
    scenes_csv_path = os.path.join(output_dir, "scenes.csv")
    
    detect_command = [
        "scenedetect",
        "--input", video_path,
        "--output", output_dir,
        "detect-content",
        "list-scenes",
        "--output", scenes_csv_path,
        "--quiet",
    ]
    
    try:
        subprocess.run(detect_command, check=True)
        return _parse_scenes_csv(scenes_csv_path)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"Error running scenedetect CLI: {e}")
        return []


def _save_scenes_csv(output_dir: str, scene_list_timecodes: List, scene_boundaries_sec: List[Tuple[float, float]]):
    """Save scene information to CSV file."""
    import csv
    
    csv_path = os.path.join(output_dir, "scenes.csv")
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Scene", "Start Frame", "End Frame", 
                "Start Time (sec)", "End Time (sec)", "Duration (sec)"
            ])
            
            for i, (start_tc, end_tc) in enumerate(scene_list_timecodes):
                start_time_sec, end_time_sec = scene_boundaries_sec[i]
                duration_sec = end_time_sec - start_time_sec
                writer.writerow([
                    i + 1,  # 1-based scene number
                    start_tc.get_frames(),
                    end_tc.get_frames(),
                    f"{start_time_sec:.3f}",
                    f"{end_time_sec:.3f}",
                    f"{duration_sec:.3f}",
                ])
        logger.info("Scene list saved to %s", csv_path)
    except Exception as e:
        logger.error(f"Error writing scene CSV: {e}")


def _parse_scenes_csv(csv_path: str) -> List[Tuple[float, float]]:
    """Parse scene boundaries from CSV file."""
    import csv
    scenes = []
    
    try:
        with open(csv_path, "r") as f:
            # Skip header lines (there may be multiple)
            for _ in range(2):
                next(f)
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 5:  # Ensure we have enough columns
                    try:
                        start_time = float(parts[3])  # Start Time (sec)
                        end_time = float(parts[4])    # End Time (sec)
                        scenes.append((start_time, end_time))
                    except (ValueError, IndexError):
                        logger.warning(f"Could not parse scene line: {line.strip()}")
                        continue
    except Exception as e:
        logger.error(f"Error parsing scenes CSV: {e}")
    
    return scenes


def extract_unique_frames_from_scenes(
    video_path: str,
    output_dir: str,
    num_frames_per_scene: int = 1,
    hash_threshold: int = 5,
    use_enhanced_detection: bool = True
) -> List[Dict]:
    """
    Enhanced scene detection and frame extraction with duplicate filtering.

    Args:
        video_path: Path to the input video file
        output_dir: Directory to save the extracted frames
        num_frames_per_scene: Number of frames to extract from each scene
        hash_threshold: Threshold for considering images as duplicates
        use_enhanced_detection: Whether to use enhanced multi-detector approach

    Returns:
        List of dictionaries with scene info and unique frame paths
    """
    os.makedirs(output_dir, exist_ok=True)
    scene_info = []
    frame_hashes = set()

    # Step 1: Detect scenes using enhanced method
    if use_enhanced_detection:
        scene_boundaries = detect_scenes_enhanced(video_path, output_dir)
    else:
        # Fallback to CLI method
        scene_boundaries = _detect_scenes_cli_fallback(video_path, output_dir)
    
    if not scene_boundaries:
        logger.warning("No scenes detected")
        return []

    # Step 2: Extract frames from each scene using enhanced extraction
    from .video_segmentation import extract_scene_keyframes
    
    frames_dir = os.path.join(output_dir, "frames")
    scene_frames = extract_scene_keyframes(
        video_path=video_path,
        scene_boundaries=scene_boundaries,
        output_dir=frames_dir,
        frames_per_scene=num_frames_per_scene
    )

    # Step 3: Filter for unique frames using image hashing
    for scene_idx, frame_paths in scene_frames.items():
        scene_data = {"scene_id": scene_idx + 1, "frame_paths": []}
        
        for frame_path in frame_paths:
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