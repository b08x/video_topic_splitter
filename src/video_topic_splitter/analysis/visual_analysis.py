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
    print(f"Analyzing video: {input_video}")
    scenes_dir = os.path.join(project_path, "scenes")
    os.makedirs(scenes_dir, exist_ok=True)

    # Use the new function to get scenes and unique frames
    scene_info = extract_unique_frames_from_scenes(
        input_video, scenes_dir, num_frames_per_scene
    )

    if not scene_info:
        logger.warning("No scenes were detected or no unique frames could be extracted.")
        return []

    print(f"Detected {len(scene_info)} scenes with unique frames to analyze.")
    save_checkpoint(
        project_path,
        CHECKPOINTS["SCENES_DETECTED"],
        {"scene_info": scene_info, "total_scenes": len(scene_info)},
    )

    analyzed_scenes = []
    for scene in progressbar.progressbar(scene_info):
        scene_id = scene["scene_id"]
        frame_analyses = []

        for frame_path in scene["frame_paths"]:
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
            except (UnidentifiedImageError, Exception) as e:
                logger.error(f"Failed to process frame {frame_path}: {e}")
                continue

        analyzed_scenes.append(
            {
                "scene_id": scene_id,
                "frame_analyses": frame_analyses,
            }
        )

    save_checkpoint(
        project_path,
        CHECKPOINTS["VISUAL_ANALYSIS_COMPLETE"],
        {"analyzed_scenes": analyzed_scenes},
    )

    return analyzed_scenes