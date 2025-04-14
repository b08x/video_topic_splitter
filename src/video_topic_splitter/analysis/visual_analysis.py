#!/usr/bin/env python3
"""Visual analysis functionalities for video scenes."""

import json
import logging
import os
from typing import List, Dict, Optional, Any

import cv2
import numpy as np
import tqdm # Using tqdm for progress bars
from PIL import Image, UnidentifiedImageError

from ..api.gemini import analyze_with_gemini
from ..constants import CHECKPOINTS
from ..processing.ocr.ocr_detection import detect_software_names # Keep OCR
# Removed: from ..processing.software.software_detection import detect_software_logos
from ..processing.video.scene_detection import extract_scene_frames # Use this for frame extraction
from ..project import save_checkpoint, load_checkpoint # Import load_checkpoint

logger = logging.getLogger(__name__)

# Removed: DEFAULT_LOGO_DB_PATH

# --- Utility Functions for Persistence (Keep as they are useful) ---

def load_analyzed_scenes(scenes_dir: str, filename: str = "scene_analysis_results.json") -> List[Dict]:
    """Loads previously analyzed scene data from a JSON file."""
    analysis_file = os.path.join(scenes_dir, filename)
    if os.path.exists(analysis_file):
        try:
            with open(analysis_file, "r", encoding="utf-8") as f:
                logger.info(f"Loading existing scene analysis from: {analysis_file}")
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load or parse existing analysis file {analysis_file}: {e}")
            return []
    else:
        logger.info(f"No existing scene analysis file found at: {analysis_file}")
        return []

def save_analyzed_scenes(scenes_dir: str, analyzed_data: List[Dict], filename: str = "scene_analysis_results.json"):
    """Saves the current state of analyzed scenes to a JSON file."""
    os.makedirs(scenes_dir, exist_ok=True)
    analysis_file = os.path.join(scenes_dir, filename)
    try:
        with open(analysis_file, "w", encoding="utf-8") as f:
            json.dump(analyzed_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved scene analysis progress to: {analysis_file}")
    except (IOError, TypeError) as e:
        logger.error(f"Failed to save analysis file {analysis_file}: {e}")


# --- Core Analysis Functions ---

def analyze_frame(
    frame_path: str,
    scene_context: Dict, # Context about the scene this frame belongs to
    software_list: Optional[List[str]] = None,
    ocr_lang: str = "eng",
    # Removed logo_db_path, logo_threshold
    previous_analysis_summary: Optional[str] = None, # Simpler context from previous frame
) -> Dict[str, Any]:
    """Analyzes a single frame image file within the context of its scene.

    Performs software detection using OCR, then uses Gemini to generate a
    textual description of the frame, incorporating scene context, OCR results,
    and optionally a summary of the previous frame's analysis.

    Args:
        frame_path (str): The file path to the frame image.
        scene_context (Dict): Dictionary containing context about the scene, e.g.,
                              {'scene_id': int, 'start_time': float, 'end_time': float}.
        software_list (Optional[List[str]], optional): List of specific software names.
        ocr_lang (str, optional): Language(s) for OCR detection. Defaults to "eng".
        previous_analysis_summary (Optional[str], optional): Text summary from the
                                                            previous frame's Gemini analysis.

    Returns:
        Dict[str, Any]: A dictionary containing the analysis results:
            'frame_path' (str): Path to the analyzed frame.
            'gemini_analysis' (str): Textual description from Gemini.
            'software_detections' (List[Dict]): OCR matches.
            'error' (str, optional): Error message if analysis failed.
    """
    logger.debug(f"Analyzing frame: {frame_path} for scene {scene_context.get('scene_id', 'N/A')}")
    results = {
        "frame_path": frame_path,
        "gemini_analysis": "Analysis Skipped",
        "software_detections": [], # Store only OCR results now
    }

    try:
        # --- Image Loading ---
        # Read with OpenCV for detection functions
        frame_cv = cv2.imread(frame_path)
        if frame_cv is None:
            raise ValueError(f"cv2.imread failed for {frame_path}")

        # Load with PIL for Gemini
        try:
            image_pil = Image.open(frame_path).convert('RGB')
        except (FileNotFoundError, UnidentifiedImageError) as e:
             raise ValueError(f"PIL could not open image file: {frame_path}. Error: {e}")

        # --- Software Detection (OCR Only) ---
        ocr_matches = []
        if software_list:
            try:
                ocr_matches = detect_software_names(frame_cv, software_list, ocr_lang)
                logger.debug(f"OCR Matches: {ocr_matches}")
                if ocr_matches:
                    results["software_detections"] = ocr_matches # Store matches
            except Exception as e:
                logger.error(f"OCR detection failed on {frame_path}: {e}")
        else:
            logger.debug("No software list provided, skipping OCR detection.")

        # --- Build Context for Gemini ---
        prompt_context_parts = [
            (f"Analyze the following frame from a video scene "
             f"(Scene ID: {scene_context.get('scene_id', 'N/A')}, "
             f"Time: {scene_context.get('start_time', 0):.2f}s - {scene_context.get('end_time', 0):.2f}s)."),
            "Describe the visual content, focusing on:",
            "1. Identifying the primary activity or user interface shown.",
            "2. Describing key visual elements (e.g., menus, buttons, code, diagrams, text content).",
            "3. Noting any specific actions or state represented in the interface."
        ]

        if ocr_matches:
            ocr_summary = ", ".join(f"{m['software']} (found text: '{m['detected_text']}')" for m in ocr_matches)
            prompt_context_parts.append(f"\nContext from Text Detection: {ocr_summary}")

        if previous_analysis_summary:
             # Limit length of previous summary
             max_prev_len = 250
             truncated_prev_summary = (previous_analysis_summary[:max_prev_len] + "...") \
                 if len(previous_analysis_summary) > max_prev_len else previous_analysis_summary
             prompt_context_parts.append(f"\nContext from Previous Frame Analysis:\n---\n{truncated_prev_summary}\n---")
             prompt_context_parts.append("\nNote any significant visual changes from the previous frame.")

        prompt_context_parts.append("\nBe concise and informative in your description.")
        full_prompt = "\n".join(prompt_context_parts)

        logger.debug(f"Gemini Prompt (frame): {full_prompt[:500]}...")

        # --- Get Gemini Analysis ---
        gemini_analysis_text = "Gemini analysis was not performed or failed."
        try:
            gemini_analysis_text = analyze_with_gemini(full_prompt, image_pil)
            logger.debug("Gemini analysis successful for frame.")
            results["gemini_analysis"] = gemini_analysis_text
        except ValueError as ve: # Catch configuration errors specifically
            logger.error(f"Gemini analysis configuration error: {ve}")
            results["gemini_analysis"] = f"Gemini analysis configuration error: {ve}"
            results["error"] = str(ve)
        except Exception as e:
            logger.error(f"Gemini analysis failed for frame {frame_path}: {e}", exc_info=True)
            results["gemini_analysis"] = f"Gemini analysis failed: {e}"
            results["error"] = str(e)

        return results

    except FileNotFoundError:
        logger.error(f"Frame file not found: {frame_path}")
        results["error"] = f"File not found: {frame_path}"
        results["gemini_analysis"] = "Analysis failed: File not found."
        return results
    except ValueError as ve: # Catch ValueErrors raised internally (e.g., image loading)
        logger.error(f"Value error analyzing frame {frame_path}: {ve}")
        results["error"] = str(ve)
        results["gemini_analysis"] = f"Analysis failed: {ve}"
        return results
    except Exception as e:
        logger.error(f"Unexpected error analyzing frame {frame_path}: {e}", exc_info=True)
        results["error"] = f"Unexpected error: {str(e)}"
        results["gemini_analysis"] = f"Analysis failed due to unexpected error: {str(e)}"
        return results


def analyze_scenes(
    input_video: str,
    scene_boundaries: List[Tuple[float, float]], # Detected scene boundaries
    project_path: str,
    software_list: Optional[List[str]] = None,
    ocr_lang: str = "eng",
    frames_per_scene: int = 1, # How many frames to analyze per scene
    frame_format: str = "jpg",
    compression_quality: int = 90,
    # Removed parameters related to logos, segment-based analysis
) -> List[Dict]:
    """Analyzes representative frames from detected video scenes.

    Orchestrates the visual analysis process for a video file based on pre-detected
    scene boundaries. It extracts representative frames for each scene and then
    analyzes them using OCR and Gemini. Manages checkpointing and saves results.

    Args:
        input_video (str): Path to the input video file.
        scene_boundaries (List[Tuple[float, float]]): List of tuples representing
            scene start and end times in seconds.
        project_path (str): Path to the project directory for saving outputs.
        software_list (Optional[List[str]], optional): List of software names for OCR.
        ocr_lang (str, optional): Language for OCR. Defaults to "eng".
        frames_per_scene (int, optional): Number of frames to extract and analyze
                                          per scene. Defaults to 1.
        frame_format (str, optional): Format for extracted frames ('jpg'/'png').
                                      Defaults to "jpg".
        compression_quality (int, optional): JPEG quality (1-100). Defaults to 90.

    Returns:
        List[Dict]: A list of dictionaries, each representing the analysis results
                    for a scene, including aggregated analysis and individual
                    frame analysis details.

    Raises:
        FileNotFoundError: If the input_video does not exist.
        RuntimeError: If a critical error occurs during processing.
        ValueError: If scene_boundaries is empty.
    """
    logger.info(f"Starting visual analysis for video: {input_video} based on {len(scene_boundaries)} scenes.")

    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video file not found: {input_video}")
    if not scene_boundaries:
         raise ValueError("Scene boundaries must be provided for analysis.")

    # --- Setup Directories and Load State ---
    os.makedirs(project_path, exist_ok=True)
    analysis_results_dir = os.path.join(project_path, "scene_analysis") # Specific dir for results
    scene_frames_dir = os.path.join(project_path, "scene_frames") # Dir for frame images
    os.makedirs(analysis_results_dir, exist_ok=True)
    os.makedirs(scene_frames_dir, exist_ok=True)

    # Load previously analyzed data to potentially resume
    analyzed_scene_results = load_analyzed_scenes(analysis_results_dir)
    processed_scene_ids = {item.get("scene_id") for item in analyzed_scene_results}
    logger.info(f"Loaded {len(analyzed_scene_results)} previously analyzed scenes.")

    # --- Extract Frames for All Scenes ---
    # Check if frames were already extracted (e.g., via checkpoint)
    # Simple check: see if scene_frames_dir contains expected subdirs or files
    # A more robust check would use checkpoints.
    frames_already_extracted = os.path.exists(os.path.join(scene_frames_dir, "1-1." + frame_format)) # Check for first frame of first scene

    all_scene_frame_info = []
    if not frames_already_extracted:
        logger.info("Extracting representative frames for scenes...")
        try:
            all_scene_frame_info = extract_scene_frames(
                video_path=input_video,
                scene_boundaries=scene_boundaries,
                output_dir=scene_frames_dir,
                num_frames_per_scene=frames_per_scene,
                frame_format=frame_format,
                jpg_quality=compression_quality,
            )
            logger.info(f"Extracted frames for {len(all_scene_frame_info)} scenes.")
            # Optional: Save checkpoint indicating frame extraction complete
        except Exception as e:
            logger.error(f"Failed to extract scene frames: {e}", exc_info=True)
            # Decide how to proceed: raise error or try to continue without frames?
            raise RuntimeError("Frame extraction failed, cannot proceed with analysis.") from e
    else:
         logger.info("Scene frames seem to exist already. Attempting to load info...")
         # Reconstruct scene_info based on existing files (less ideal than checkpoint)
         # This logic assumes files follow the '$SCENE_NUMBER-$IMAGE_NUMBER.ext' template
         for i, (start_time, end_time) in enumerate(scene_boundaries):
             scene_id = i + 1
             frame_paths = []
             for frame_num in range(1, frames_per_scene + 1):
                 expected_path = os.path.join(scene_frames_dir, f"{scene_id}-{frame_num}.{frame_format}")
                 if os.path.exists(expected_path):
                     frame_paths.append(expected_path)
             if frame_paths: # Only add if frames were found
                 all_scene_frame_info.append({
                     "scene_id": scene_id,
                     "start_time": start_time,
                     "end_time": end_time,
                     "duration": end_time - start_time,
                     "frame_paths": frame_paths,
                 })
         logger.info(f"Reconstructed frame info for {len(all_scene_frame_info)} scenes from existing files.")


    if not all_scene_frame_info:
         logger.error("No frame information available (extraction failed or no frames found). Cannot perform analysis.")
         return analyzed_scene_results # Return whatever was loaded

    # --- Analyze Each Scene Frame ---
    logger.info("Analyzing frames from detected scenes...")

    with tqdm.tqdm(total=len(all_scene_frame_info), desc="Analyzing Scenes", unit="scene") as pbar:
        for scene_info in all_scene_frame_info:
            scene_id = scene_info.get("scene_id")

            # Skip if already processed
            if scene_id in processed_scene_ids:
                logger.debug(f"Skipping scene {scene_id} (already processed).")
                pbar.update(1)
                continue

            logger.info(f"Processing scene {scene_id} ({scene_info['start_time']:.2f}s - {scene_info['end_time']:.2f}s)")
            scene_frame_analyses = []
            error_in_scene = False
            previous_gemini_summary = None

            scene_context = {
                 "scene_id": scene_id,
                 "start_time": scene_info["start_time"],
                 "end_time": scene_info["end_time"],
            }

            for frame_path in sorted(scene_info.get("frame_paths", [])): # Process frames in order
                if not os.path.exists(frame_path):
                    logger.warning(f"Frame path not found: {frame_path}. Skipping.")
                    continue

                try:
                    # Analyze the individual frame
                    frame_analysis_result = analyze_frame(
                        frame_path=frame_path,
                        scene_context=scene_context,
                        software_list=software_list,
                        ocr_lang=ocr_lang,
                        previous_analysis_summary=previous_gemini_summary,
                    )
                    scene_frame_analyses.append(frame_analysis_result)

                    if "error" in frame_analysis_result:
                        error_in_scene = True # Mark if any frame analysis failed
                    else:
                        # Update context for the next frame in this scene
                        previous_gemini_summary = frame_analysis_result.get("gemini_analysis")

                except Exception as frame_err:
                    logger.error(f"Error analyzing frame {frame_path} for scene {scene_id}: {frame_err}", exc_info=True)
                    scene_frame_analyses.append({
                        "frame_path": frame_path,
                        "error": f"Failed to analyze frame: {frame_err}",
                        "gemini_analysis": "Analysis failed.",
                        "software_detections": []
                    })
                    error_in_scene = True

            # Aggregate results for the scene
            # Combine Gemini summaries for an overall scene description
            combined_summary = " ".join(
                res.get("gemini_analysis", "") for res in scene_frame_analyses if "error" not in res
            ).strip()
            # Basic aggregation of software detections
            all_ocr = [match for res in scene_frame_analyses for match in res.get("software_detections", [])]
            unique_software = sorted(list(set(match['software'] for match in all_ocr)))


            scene_analysis_result = {
                "scene_id": scene_id,
                "start_time": scene_info["start_time"],
                "end_time": scene_info["end_time"],
                "duration": scene_info.get("duration", scene_info["end_time"] - scene_info["start_time"]),
                "combined_summary": combined_summary, # Add combined summary
                "detected_software": unique_software, # List unique software names
                "frame_analyses": scene_frame_analyses, # Keep individual frame details
                "analysis_errors": error_in_scene,
            }

            # Add to overall results and save progress
            analyzed_scene_results.append(scene_analysis_result)
            save_analyzed_scenes(analysis_results_dir, analyzed_scene_results) # Save after each scene

            # Save scene-specific checkpoint (optional but good for recovery)
            # save_checkpoint(
            #     project_path, f"SCENE_{scene_id}_ANALYZED", {"scene_id": scene_id, "analysis": scene_analysis_result}
            # )
            pbar.update(1)

    logger.info("Scene analysis complete.")

    # --- Final Checkpoint ---
    logger.info("Saving final scene analysis checkpoint...")
    final_checkpoint_data = {
        "total_scenes_processed": len(analyzed_scene_results),
        "results_path": os.path.join(analysis_results_dir, "scene_analysis_results.json"),
        "scene_frames_dir": scene_frames_dir,
    }
    save_checkpoint(
        project_path,
        CHECKPOINTS["SCENE_ANALYSIS_COMPLETE"],
        final_checkpoint_data,
    )

    return analyzed_scene_results

