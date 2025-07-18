#!/usr/bin/env python3
"""Visual analysis functionalities, including OCR based software detection."""
import json
import logging
import os

import cv2
import progressbar
from PIL import Image, UnidentifiedImageError

from ..api.gemini import analyze_with_gemini
from ..constants import CHECKPOINTS
from ..processing.ocr.ocr_detection import detect_software_names
from ..processing.video.scene_detection import extract_unique_frames_from_scenes
from ..project import save_checkpoint
from ..progress_tracker import ProgressTracker, SubProgressTracker

logger = logging.getLogger(__name__)


def analyze_screenshot(
    image_path,
    project_path,
    software_list=None,
    ocr_lang="eng",
    context=None,
):
    """Analyze a single screenshot for software applications using OCR."""
    # ... (existing screenshot analysis logic remains the same)


def split_and_analyze_video(
    input_video,
    project_path,
    software_list=None,
    ocr_lang="eng",
    frames_per_scene=1,
    register="it-workflow",
    progress_tracker: ProgressTracker = None,
):
    """
    Splits a video into scenes, extracts unique frames, and analyzes them.

    This function uses PySceneDetect to identify scenes and extract a specified
    number of unique frames from each. It then uses Gemini for visual analysis
    and OCR for software detection on each frame.

    Args:
        input_video: Path to the input video file.
        project_path: Path to the project directory.
        software_list: Optional list of software names to detect via OCR.
        ocr_lang: Language for OCR detection.
        frames_per_scene: Number of unique frames to extract per scene.
        register: The analysis register (e.g., 'it-workflow') for Gemini.

    Returns:
        A list of dictionaries, where each dictionary contains the analysis
        for a single scene.
    """
    if progress_tracker:
        progress_tracker.start_phase("Visual Analysis")
        progress_tracker.update_phase_progress(0.0, "Setting up scene detection...")
    else:
        print(f"Analyzing video: {input_video}")
    
    scenes_dir = os.path.join(project_path, "scenes")
    os.makedirs(scenes_dir, exist_ok=True)

    # Use the enhanced function to get scenes and unique frames
    if progress_tracker:
        progress_tracker.update_phase_progress(10.0, "Extracting scenes and unique frames...")
    scene_info = extract_unique_frames_from_scenes(
        input_video, scenes_dir, frames_per_scene, use_enhanced_detection=True
    )

    if not scene_info:
        logger.warning("No scenes were detected or no unique frames could be extracted.")
        if progress_tracker:
            progress_tracker.fail_phase("No scenes detected or no unique frames extracted")
        return []

    if progress_tracker:
        progress_tracker.update_phase_progress(20.0, f"Detected {len(scene_info)} scenes with unique frames to analyze")
    else:
        print(f"Detected {len(scene_info)} scenes with unique frames to analyze.")
    
    save_checkpoint(
        project_path,
        CHECKPOINTS["SCENES_DETECTED"],
        {"scene_info": scene_info, "total_scenes": len(scene_info)},
    )

    # Create progress tracker for scene analysis
    scene_descriptions = [f"Scene {scene['scene_id']}" for scene in scene_info]
    sub_tracker = None
    if progress_tracker:
        sub_tracker = progress_tracker.create_sub_progress_tracker("Visual Analysis", scene_descriptions)
    
    analyzed_scenes = []
    scene_iterator = progressbar.progressbar(scene_info) if not progress_tracker else scene_info
    for scene_idx, scene in enumerate(scene_iterator):
        scene_id = scene["scene_id"]
        frame_analyses = []
        
        if sub_tracker:
            sub_tracker.update_item_progress(0.0, f"Analyzing scene {scene_id}")
        
        total_frames = len(scene["frame_paths"])
        for frame_idx, frame_path in enumerate(scene["frame_paths"]):
            try:
                frame = cv2.imread(frame_path)
                if frame is None:
                    logger.warning(f"Could not read frame: {frame_path}")
                    continue

                image = Image.open(frame_path)

                # Perform OCR and Gemini analysis
                ocr_matches = detect_software_names(frame, software_list, ocr_lang)
                software_context = (
                    f"Detected software (via OCR): {', '.join(m['software'] for m in ocr_matches)}"
                    if ocr_matches
                    else "No specific software detected via OCR."
                )

                prompt = (
                    f"Analyze this frame from scene {scene_id}. "
                    f"The overall topic register is '{register}'.\n"
                    f"{software_context}\n\n"
                    "Describe the visual elements, user interface components, and any actions taking place."
                )

                gemini_analysis = analyze_with_gemini(prompt, image)

                frame_analyses.append(
                    {
                        "frame_path": frame_path,
                        "ocr_matches": ocr_matches,
                        "gemini_analysis": gemini_analysis,
                    }
                )
                
                if sub_tracker:
                    frame_progress = ((frame_idx + 1) / total_frames) * 100.0
                    sub_tracker.update_item_progress(frame_progress, f"Processed frame {frame_idx + 1}/{total_frames}")
            except (UnidentifiedImageError, Exception) as e:
                logger.error(f"Failed to process frame {frame_path}: {e}")
                continue

        analyzed_scenes.append(
            {
                "scene_id": scene_id,
                "frame_analyses": frame_analyses,
            }
        )
        
        if sub_tracker:
            sub_tracker.complete_item()

    save_checkpoint(
        project_path,
        CHECKPOINTS["VISUAL_ANALYSIS_COMPLETE"],
        {"analyzed_scenes": analyzed_scenes},
    )
    
    if progress_tracker:
        progress_tracker.complete_phase("Visual Analysis")

    return analyzed_scenes