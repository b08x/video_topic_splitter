#!/usr/bin/env python3
"""Visual analysis functionalities for video scenes."""

import logging
import os
import cv2
from typing import Dict, List, Tuple, Any, Optional
import tqdm
from PIL import Image, UnidentifiedImageError

# Import the new client class
from ..api.gemini import GeminiClient
from ..constants import CHECKPOINTS
from ..processing.ocr.ocr_detection import detect_software_names
from ..processing.video.scene_detection import extract_scene_frames
from ..project import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)

# --- Persistence Functions ---

def load_analyzed_scenes(analysis_dir: str) -> List[Dict]:
    """Load previously analyzed scene results from disk."""
    results_path = os.path.join(analysis_dir, "scene_analysis_results.json")
    if os.path.exists(results_path):
        try:
            import json
            with open(results_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load scene analysis results: {e}")
    return []

def save_analyzed_scenes(analysis_dir: str, scene_results: List[Dict]) -> None:
    """Save analyzed scene results to disk."""
    results_path = os.path.join(analysis_dir, "scene_analysis_results.json")
    try:
        import json
        with open(results_path, 'w') as f:
            json.dump(scene_results, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save scene analysis results: {e}")


# --- Core Analysis Functions ---

def analyze_frame(
    frame_path: str,
    scene_context: Dict,
    gemini_client: GeminiClient, # Add client as parameter
    software_list: Optional[List[str]] = None,
    ocr_lang: str = "eng",
    previous_analysis_summary: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyzes a single frame image file within the context of its scene."""
    logger.debug(f"Analyzing frame: {frame_path} for scene {scene_context.get('scene_id', 'N/A')}")
    results = {
        "frame_path": frame_path,
        "gemini_analysis": "Analysis Skipped",
        "software_detections": [],
    }

    try:
        # --- Image Loading ---
        frame_cv = cv2.imread(frame_path)
        if frame_cv is None:
            raise ValueError(f"cv2.imread failed for {frame_path}")
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
                    results["software_detections"] = ocr_matches
            except Exception as e:
                logger.error(f"OCR detection failed on {frame_path}: {e}")
        else:
            logger.debug("No software list provided, skipping OCR detection.")

        # --- Build Context for Gemini ---
        # (Prompt building logic remains the same as in previous refactor step)
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
              max_prev_len = 250
              truncated_prev_summary = (previous_analysis_summary[:max_prev_len] + "...") \
                  if len(previous_analysis_summary) > max_prev_len else previous_analysis_summary
              prompt_context_parts.append(f"\nContext from Previous Frame Analysis:\n---\n{truncated_prev_summary}\n---")
              prompt_context_parts.append("\nNote any significant visual changes from the previous frame.")
        prompt_context_parts.append("\nBe concise and informative in your description.")
        full_prompt = "\n".join(prompt_context_parts)

        logger.debug(f"Gemini Prompt (frame): {full_prompt[:500]}...")

        # --- Get Gemini Analysis using the client's method ---
        gemini_analysis_text = gemini_client.analyze(full_prompt, image_pil) # Call the class method
        results["gemini_analysis"] = gemini_analysis_text
        if "Analysis failed" in gemini_analysis_text: # Check if the returned string indicates failure
             results["error"] = gemini_analysis_text # Store error message if analysis failed

        return results

    # ... (Error handling remains similar) ...
    except FileNotFoundError:
        logger.error(f"Frame file not found: {frame_path}")
        results["error"] = f"File not found: {frame_path}"
        results["gemini_analysis"] = "Analysis failed: File not found."
        return results
    except ValueError as ve:
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
    scene_boundaries: List[Tuple[float, float]],
    project_path: str,
    gemini_client: GeminiClient, # Pass the client instance here
    software_list: Optional[List[str]] = None,
    ocr_lang: str = "eng",
    frames_per_scene: int = 1,
    frame_format: str = "jpg",
    compression_quality: int = 90,
    register: str = "it-workflow", # Keep register for potential future prompt adjustments
) -> List[Dict]:
    """Analyzes representative frames from detected video scenes."""
    # ... (Setup and frame extraction logic remains the same as previous refactor step) ...
    logger.info(f"Starting visual analysis for video: {input_video} based on {len(scene_boundaries)} scenes.")

    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video file not found: {input_video}")
    if not scene_boundaries:
         raise ValueError("Scene boundaries must be provided for analysis.")

    analysis_results_dir = os.path.join(project_path, "scene_analysis")
    scene_frames_dir = os.path.join(project_path, "scene_frames")
    os.makedirs(analysis_results_dir, exist_ok=True)
    os.makedirs(scene_frames_dir, exist_ok=True)

    analyzed_scene_results = load_analyzed_scenes(analysis_results_dir)
    processed_scene_ids = {item.get("scene_id") for item in analyzed_scene_results}
    logger.info(f"Loaded {len(analyzed_scene_results)} previously analyzed scenes.")

    all_scene_frame_info = []
    # Logic to extract or load frame info (same as previous step)
    # ... (assuming all_scene_frame_info gets populated correctly) ...
    frames_already_extracted = os.path.exists(os.path.join(scene_frames_dir, "1-1." + frame_format))
    if not frames_already_extracted:
        logger.info("Extracting representative frames for scenes...")
        try:
            all_scene_frame_info = extract_scene_frames(
                video_path=input_video, scene_boundaries=scene_boundaries, output_dir=scene_frames_dir,
                num_frames_per_scene=frames_per_scene, frame_format=frame_format, jpg_quality=compression_quality,
            )
            logger.info(f"Extracted frames for {len(all_scene_frame_info)} scenes.")
        except Exception as e:
            logger.error(f"Failed to extract scene frames: {e}", exc_info=True)
            raise RuntimeError("Frame extraction failed, cannot proceed with analysis.") from e
    else:
         logger.info("Scene frames seem to exist already. Attempting to load info...")
         for i, (start_time, end_time) in enumerate(scene_boundaries):
             scene_id = i + 1; frame_paths = []
             for frame_num in range(1, frames_per_scene + 1):
                 expected_path = os.path.join(scene_frames_dir, f"{scene_id}-{frame_num}.{frame_format}")
                 if os.path.exists(expected_path): frame_paths.append(expected_path)
             if frame_paths:
                 all_scene_frame_info.append({
                     "scene_id": scene_id, "start_time": start_time, "end_time": end_time,
                     "duration": end_time - start_time, "frame_paths": frame_paths,
                 })
         logger.info(f"Reconstructed frame info for {len(all_scene_frame_info)} scenes.")

    if not all_scene_frame_info:
         logger.error("No frame information available. Cannot perform analysis.")
         return analyzed_scene_results

    logger.info("Analyzing frames from detected scenes...")
    with tqdm.tqdm(total=len(all_scene_frame_info), desc="Analyzing Scenes", unit="scene") as pbar:
        for scene_info in all_scene_frame_info:
            scene_id = scene_info.get("scene_id")
            if scene_id in processed_scene_ids:
                logger.debug(f"Skipping scene {scene_id} (already processed).")
                pbar.update(1); continue

            logger.info(f"Processing scene {scene_id} ({scene_info['start_time']:.2f}s - {scene_info['end_time']:.2f}s)")
            scene_frame_analyses = []
            error_in_scene = False
            previous_gemini_summary = None
            scene_context = {"scene_id": scene_id, "start_time": scene_info["start_time"], "end_time": scene_info["end_time"]}

            for frame_path in sorted(scene_info.get("frame_paths", [])):
                if not os.path.exists(frame_path):
                    logger.warning(f"Frame path not found: {frame_path}. Skipping."); continue

                try:
                    # Pass the instantiated gemini_client to analyze_frame
                    frame_analysis_result = analyze_frame(
                        frame_path=frame_path,
                        scene_context=scene_context,
                        gemini_client=gemini_client, # Pass client here
                        software_list=software_list,
                        ocr_lang=ocr_lang,
                        previous_analysis_summary=previous_gemini_summary,
                    )
                    scene_frame_analyses.append(frame_analysis_result)
                    if "error" in frame_analysis_result: error_in_scene = True
                    else: previous_gemini_summary = frame_analysis_result.get("gemini_analysis")
                except Exception as frame_err:
                    # ... (error handling for frame analysis call) ...
                    logger.error(f"Error analyzing frame {frame_path} for scene {scene_id}: {frame_err}", exc_info=True)
                    scene_frame_analyses.append({
                        "frame_path": frame_path, "error": f"Failed to analyze frame: {frame_err}",
                        "gemini_analysis": "Analysis failed.", "software_detections": []
                    })
                    error_in_scene = True


            # --- Aggregate results (remains the same) ---
            combined_summary = " ".join(res.get("gemini_analysis", "") for res in scene_frame_analyses if "error" not in res).strip()
            all_ocr = [match for res in scene_frame_analyses for match in res.get("software_detections", [])]
            unique_software = sorted(list(set(match['software'] for match in all_ocr)))
            scene_analysis_result = {
                "scene_id": scene_id, "start_time": scene_info["start_time"], "end_time": scene_info["end_time"],
                "duration": scene_info.get("duration", scene_info["end_time"] - scene_info["start_time"]),
                "combined_summary": combined_summary, "detected_software": unique_software,
                "frame_analyses": scene_frame_analyses, "analysis_errors": error_in_scene,
            }
            analyzed_scene_results.append(scene_analysis_result)
            save_analyzed_scenes(analysis_results_dir, analyzed_scene_results)
            pbar.update(1)

    logger.info("Scene analysis complete.")
    # --- Final Checkpoint (remains the same) ---
    logger.info("Saving final scene analysis checkpoint...")
    final_checkpoint_data = {
        "total_scenes_processed": len(analyzed_scene_results),
        "results_path": os.path.join(analysis_results_dir, "scene_analysis_results.json"),
        "scene_frames_dir": scene_frames_dir,
    }
    save_checkpoint(project_path, CHECKPOINTS["SCENE_ANALYSIS_COMPLETE"], final_checkpoint_data)

    return analyzed_scene_results
