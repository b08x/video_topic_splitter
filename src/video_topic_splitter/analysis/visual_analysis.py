#!/usr/bin/env python3
"""Visual analysis functionalities for videos and screenshots.

This module provides functions to perform visual analysis, primarily focused on:
1.  Analyzing individual screenshots: Detecting software via OCR and logo matching,
    and generating a textual description using a multimodal LLM (Gemini).
2.  Analyzing videos segment by segment: Either based on pre-defined transcript
    segments or by detecting scene changes using PySceneDetect. For each segment
    or scene, it extracts representative frames, analyzes them (similar to
    screenshot analysis), and aggregates the findings.

It integrates components for OCR, logo detection, scene detection, frame
extraction/quality assessment (via `ContextualFrameAnalyzer`), and interaction
with the Gemini API. It also includes checkpointing and result persistence.
"""
import json
import logging
import os
from typing import List, Dict, Optional, Any # Added Any for flexibility

import cv2
import numpy as np
import progressbar # Consider replacing with tqdm for consistency if preferred
from moviepy.editor import VideoFileClip
from PIL import Image, UnidentifiedImageError

from ..api.gemini import analyze_with_gemini
from ..constants import CHECKPOINTS
from ..processing.ocr.ocr_detection import detect_software_names
from ..processing.software.software_detection import detect_software_logos
from ..processing.video.scene_detection import extract_scenes_from_video
from ..project import save_checkpoint, load_checkpoint # Import load_checkpoint
from ..prompt_templates import get_analysis_prompt # Although not used directly here, keep if planned
from .frame_analysis import ContextualFrameAnalyzer

logger = logging.getLogger(__name__)

# --- Configuration ---
# Consider moving paths to a config file or constants module if they grow
# Default path assumes 'data/logos' relative to the package structure
# Adjust this logic if your structure differs.
try:
    # Assumes visual_analysis.py is in src/video_topic_splitter/analysis/
    _MODULE_DIR = os.path.dirname(__file__)
    _SRC_DIR = os.path.dirname(os.path.dirname(_MODULE_DIR))
    DEFAULT_LOGO_DB_PATH = os.path.join(_SRC_DIR, "data", "logos")
    if not os.path.isdir(DEFAULT_LOGO_DB_PATH):
        logger.warning(f"Default logo DB path not found: {DEFAULT_LOGO_DB_PATH}")
        DEFAULT_LOGO_DB_PATH = None
except Exception:
    logger.warning("Could not determine default logo DB path.")
    DEFAULT_LOGO_DB_PATH = None


# --- Utility Functions for Persistence ---

def load_analyzed_segments(segments_dir: str, filename: str = "visual_analysis_results.json") -> List[Dict]:
    """Loads previously analyzed segment/scene data from a JSON file.

    Args:
        segments_dir (str): The directory where the analysis results file is stored.
        filename (str, optional): The name of the JSON file containing the results.
                                  Defaults to "visual_analysis_results.json".

    Returns:
        List[Dict]: A list of dictionaries, each representing an analyzed segment
                    or scene. Returns an empty list if the file doesn't exist or
                    cannot be parsed.
    """
    analysis_file = os.path.join(segments_dir, filename)
    if os.path.exists(analysis_file):
        try:
            with open(analysis_file, "r", encoding="utf-8") as f:
                logger.info(f"Loading existing visual analysis from: {analysis_file}")
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load or parse existing analysis file {analysis_file}: {e}")
            return [] # Return empty list on error
    else:
        logger.info(f"No existing visual analysis file found at: {analysis_file}")
        return []


def save_analyzed_segments(segments_dir: str, analyzed_data: List[Dict], filename: str = "visual_analysis_results.json"):
    """Saves the current state of analyzed segments/scenes to a JSON file.

    Args:
        segments_dir (str): The directory where the analysis results file should be saved.
        analyzed_data (List[Dict]): The list of analyzed segment/scene dictionaries to save.
        filename (str, optional): The name of the JSON file to save the results to.
                                  Defaults to "visual_analysis_results.json".
    """
    os.makedirs(segments_dir, exist_ok=True) # Ensure directory exists
    analysis_file = os.path.join(segments_dir, filename)
    try:
        with open(analysis_file, "w", encoding="utf-8") as f:
            json.dump(analyzed_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved visual analysis progress to: {analysis_file}")
    except (IOError, TypeError) as e:
        logger.error(f"Failed to save analysis file {analysis_file}: {e}")


# --- Core Analysis Functions ---

def analyze_screenshot(
    image_path: str,
    project_path: str,
    software_list: Optional[List[str]] = None,
    logo_db_path: Optional[str] = DEFAULT_LOGO_DB_PATH,
    ocr_lang: str = "eng",
    logo_threshold: float = 0.8,
    context: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyzes a single screenshot image file.

    Performs software detection using OCR and logo matching, then uses Gemini
    to generate a textual description of the screenshot, incorporating the
    detection results and any provided context. Saves the analysis results
    and updates project checkpoints.

    Args:
        image_path (str): The file path to the screenshot image.
        project_path (str): Path to the project directory for saving results
                            and checkpoints.
        software_list (Optional[List[str]], optional): A list of specific software
            names to look for. Defaults to None (no specific filtering).
        logo_db_path (Optional[str], optional): Path to the directory containing
            logo images for detection. Defaults to `DEFAULT_LOGO_DB_PATH`.
        ocr_lang (str, optional): Language(s) for OCR detection. Defaults to "eng".
        logo_threshold (float, optional): Minimum confidence for logo detection.
            Defaults to 0.8.
        context (Optional[str], optional): Additional textual context to provide
            to the Gemini model during analysis. Defaults to None.

    Returns:
        Dict[str, Any]: A dictionary containing the analysis results:
            'gemini_analysis' (str): Textual description from Gemini.
            'software_detections' (List[Dict]): List of detected software,
                including source ('screenshot'), OCR matches, and logo matches.
                Returns an empty list if no software is detected or analysis fails.
            May contain an 'error' key if analysis failed.
    """
    logger.info(f"Analyzing screenshot: {image_path}")
    os.makedirs(project_path, exist_ok=True) # Ensure project path exists

    # Basic checkpoint management (consider moving to a dedicated project manager class)
    # if not load_checkpoint(project_path): # Check if *any* checkpoint exists
    #     save_checkpoint(
    #         project_path, CHECKPOINTS["PROJECT_CREATED"], {"project_path": project_path}
    #     )

    try:
        # --- Image Loading ---
        # Read with OpenCV for detection functions
        frame = cv2.imread(image_path)
        if frame is None:
            # Try loading with PIL as a fallback, then convert
            try:
                logger.warning(f"cv2.imread failed for {image_path}. Trying PIL.")
                pil_img = Image.open(image_path).convert('RGB') # Ensure RGB
                frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                logger.info("Successfully loaded image using PIL.")
            except (FileNotFoundError, UnidentifiedImageError, Exception) as pil_err:
                raise ValueError(f"Could not read image file with cv2 or PIL: {image_path}. Error: {pil_err}")

        # Load with PIL for Gemini (already tried above if cv2 failed)
        try:
            image = Image.open(image_path)
            # It's good practice to ensure the image mode is suitable for Gemini if known
            # image = image.convert('RGB')
        except (FileNotFoundError, UnidentifiedImageError) as e:
             raise ValueError(f"PIL could not open image file: {image_path}. Error: {e}")


        # --- Software Detection ---
        ocr_matches = []
        logo_matches = []
        if software_list: # Only run detection if a list is provided
            try:
                ocr_matches = detect_software_names(frame, software_list, ocr_lang)
                logger.debug(f"OCR Matches: {ocr_matches}")
            except Exception as e:
                logger.error(f"OCR detection failed on {image_path}: {e}")
            try:
                logo_matches = detect_software_logos(
                    frame, software_list, logo_db_path, logo_threshold
                )
                logger.debug(f"Logo Matches: {logo_matches}")
            except Exception as e:
                logger.error(f"Logo detection failed on {image_path}: {e}")
        else:
            logger.info("No software list provided, skipping OCR/Logo detection.")

        detection_analysis = {
            "ocr_matches": ocr_matches,
            "logo_matches": logo_matches,
        }

        # --- Build Context for Gemini ---
        software_context_str = ""
        if software_list and (ocr_matches or logo_matches):
            software_context_str = "\n\nSoftware Detection Context:"
            if ocr_matches:
                software_context_str += "\n- Text-based detections: " + ", ".join(
                    f"{m['software']} (found text: '{m['detected_text']}')" for m in ocr_matches
                )
            if logo_matches:
                software_context_str += "\n- Logo-based detections: " + ", ".join(
                    f"{m['software']} (confidence: {m['confidence']:.2f})" for m in logo_matches
                )

        # --- Generate Gemini Prompt ---
        # Base prompt instructing the model
        base_prompt = (
            "Analyze the provided screenshot image. Describe the visual content, focusing on: "
            "1. Identifying the primary software application or user interface shown. "
            "2. Describing key visual elements (e.g., menus, buttons, code, diagrams, text content). "
            "3. Noting any specific actions or state represented in the interface. "
            "Relate your description to the software detection context if provided."
        )

        # Combine context and base prompt
        full_prompt = base_prompt
        if software_context_str:
            full_prompt += software_context_str
        if context: # Prepend user-provided context if available
            full_prompt = f"User Context:\n{context}\n\n{full_prompt}"

        logger.debug(f"Gemini Prompt (screenshot): {full_prompt[:500]}...") # Log truncated prompt

        # --- Get Gemini Analysis ---
        gemini_analysis_text = "Gemini analysis was not performed or failed."
        try:
            gemini_analysis_text = analyze_with_gemini(full_prompt, image)
            logger.info("Gemini analysis successful for screenshot.")
        except ValueError as ve: # Catch configuration errors specifically
            logger.error(f"Gemini analysis configuration error: {ve}")
            gemini_analysis_text = f"Gemini analysis configuration error: {ve}"
        except Exception as e:
            logger.error(f"Gemini analysis failed for screenshot {image_path}: {e}", exc_info=True)
            gemini_analysis_text = f"Gemini analysis failed: {e}"

        # --- Prepare and Save Results ---
        results = {
            "image_path": image_path, # Include path for reference
            "gemini_analysis": gemini_analysis_text,
            "software_detections": (
                [{"source": "screenshot", **detection_analysis}]
                if (ocr_matches or logo_matches)
                else []
            ),
        }

        # Save results to a dedicated file for this screenshot? Or append to a main file?
        # For simplicity, let's save to a main project results file, overwriting previous.
        # Consider a more robust result management strategy for multiple analyses.
        results_path = os.path.join(project_path, "screenshot_analysis_results.json")
        try:
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Screenshot analysis results saved to: {results_path}")
        except (IOError, TypeError) as e:
            logger.error(f"Failed to save screenshot results to {results_path}: {e}")


        # --- Save Checkpoints ---
        # Checkpoint after successful analysis
        save_checkpoint(
            project_path, CHECKPOINTS["SCREENSHOT_ANALYZED"], {"results_path": results_path}
        )
        # Consider if PROCESS_COMPLETE is appropriate here or after video analysis
        # save_checkpoint(
        #     project_path, CHECKPOINTS["PROCESS_COMPLETE"], {"final_results": results_path}
        # )

        return results

    except FileNotFoundError:
        logger.error(f"Screenshot file not found: {image_path}")
        return {"error": f"File not found: {image_path}", "gemini_analysis": "", "software_detections": []}
    except ValueError as ve: # Catch ValueErrors raised internally (e.g., image loading)
        logger.error(f"Value error analyzing screenshot {image_path}: {ve}")
        return {"error": str(ve), "gemini_analysis": "", "software_detections": []}
    except Exception as e:
        logger.error(f"Unexpected error analyzing screenshot {image_path}: {e}", exc_info=True)
        return {
            "error": f"Unexpected error: {str(e)}",
            "gemini_analysis": f"Analysis failed due to unexpected error: {str(e)}",
            "software_detections": [],
        }


def split_and_analyze_video(
    input_video: str,
    segments: List[Dict], # Topic/Transcript segments
    project_path: str,
    software_list: Optional[List[str]] = None,
    logo_db_path: Optional[str] = DEFAULT_LOGO_DB_PATH,
    ocr_lang: str = "eng",
    logo_threshold: float = 0.8,
    quality_threshold: float = 0.5,
    save_format: str = "jpg",
    compression_quality: int = 85,
    extract_scenes: bool = False,
    min_scene_len: float = 1.0, # In seconds
    scene_detection_threshold: float = 27.0, # PySceneDetect threshold
    frames_per_scene: int = 1,
    register: str = "it-workflow", # Used by ContextualFrameAnalyzer prompt generation
) -> List[Dict]:
    """Analyzes a video either by transcript segments or detected scenes.

    Orchestrates the visual analysis process for a video file. It can operate
    in two modes:
    1. Segment-based: Uses the provided `segments` (from transcript/topic analysis)
       to define analysis boundaries. It leverages `ContextualFrameAnalyzer` to
       extract, analyze, and summarize frames within each segment.
    2. Scene-based: Uses `extract_scenes_from_video` (PySceneDetect) to identify
       scene changes. It then analyzes a specified number of frames (`frames_per_scene`)
       from each detected scene using OCR, logo detection, and Gemini.

    Manages checkpointing to allow resuming analysis and saves the final aggregated
    visual analysis results.

    Args:
        input_video (str): Path to the input video file.
        segments (List[Dict]): List of transcript/topic segments. Required if
            `extract_scenes` is False. Each segment dict needs 'start_time',
            'end_time', 'segment_id', 'transcript', 'dominant_topic', 'top_keywords'.
        project_path (str): Path to the project directory for saving outputs
            (screenshots, scene frames, results, checkpoints).
        software_list (Optional[List[str]], optional): List of software names for
            detection. Defaults to None.
        logo_db_path (Optional[str], optional): Path to logo database.
            Defaults to `DEFAULT_LOGO_DB_PATH`.
        ocr_lang (str, optional): Language for OCR. Defaults to "eng".
        logo_threshold (float, optional): Confidence threshold for logo detection.
            Defaults to 0.8.
        quality_threshold (float, optional): Minimum quality for saving screenshots
            (only used in segment-based mode). Defaults to 0.5.
        save_format (str, optional): Format for saved screenshots ('jpg'/'png')
            (only used in segment-based mode). Defaults to "jpg".
        compression_quality (int, optional): JPEG quality (1-100) for saved frames/
            screenshots. Defaults to 85.
        extract_scenes (bool, optional): If True, use scene detection instead of
            the provided `segments`. Defaults to False.
        min_scene_len (float, optional): Minimum scene length in seconds (used only
            if `extract_scenes` is True). Defaults to 1.0.
        scene_detection_threshold (float, optional): Threshold for PySceneDetect's
            content detector (used only if `extract_scenes` is True). Defaults to 27.0.
        frames_per_scene (int, optional): Number of frames to extract and analyze
            per detected scene (used only if `extract_scenes` is True). Defaults to 1.
        register (str, optional): Analysis register/domain passed to
            `ContextualFrameAnalyzer` for prompt generation (used only in
            segment-based mode). Defaults to "it-workflow".

    Returns:
        List[Dict]: A list of dictionaries, where each dictionary represents the
            visual analysis results for a segment or a scene. Structure depends
            on the mode (segment or scene based).

    Raises:
        FileNotFoundError: If the input_video does not exist.
        RuntimeError: If a critical error occurs during processing that prevents completion.
        ValueError: If required arguments are missing (e.g., segments when not using scene detection).
    """
    logger.info(f"Starting visual analysis for video: {input_video}")
    logger.info(f"Project path: {project_path}")
    logger.info(f"Mode: {'Scene Detection' if extract_scenes else 'Segment-based'}")

    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video file not found: {input_video}")
    if not extract_scenes and not segments:
         raise ValueError("Segments must be provided when extract_scenes is False.")

    # --- Setup Directories and Load State ---
    os.makedirs(project_path, exist_ok=True)
    # Use a consistent directory for results, regardless of mode
    results_dir = os.path.join(project_path, "visual_analysis")
    os.makedirs(results_dir, exist_ok=True)

    # Load previously analyzed data to potentially resume
    analyzed_results = load_analyzed_segments(results_dir) # Load from results_dir
    processed_ids = {item.get("segment_id") if not extract_scenes else item.get("scene_id")
                     for item in analyzed_results}
    logger.info(f"Loaded {len(analyzed_results)} previously analyzed items.")

    frame_analyzer = None # Initialize variable

    try:
        if extract_scenes:
            # --- Scene Detection Mode ---
            logger.info("Initiating scene detection...")
            scenes_output_dir = os.path.join(project_path, "scene_frames")
            os.makedirs(scenes_output_dir, exist_ok=True)

            # Detect scenes and extract representative frames
            # Check if scene detection was already completed via checkpoint
            scene_checkpoint_data = load_checkpoint(project_path, CHECKPOINTS["SCENES_DETECTED"])
            if scene_checkpoint_data and 'scene_info' in scene_checkpoint_data:
                 scene_info = scene_checkpoint_data['scene_info']
                 logger.info(f"Loaded scene info from checkpoint: {len(scene_info)} scenes.")
            else:
                 scene_info = extract_scenes_from_video(
                     video_path=input_video,
                     output_dir=scenes_output_dir,
                     min_scene_len_sec=min_scene_len,
                     threshold=scene_detection_threshold,
                     num_frames_per_scene=frames_per_scene,
                     save_format=save_format, # Use save_format here too
                     jpg_quality=compression_quality,
                 )
                 logger.info(f"Detected {len(scene_info)} scenes.")
                 # Save checkpoint after detection
                 save_checkpoint(
                     project_path,
                     CHECKPOINTS["SCENES_DETECTED"],
                     {"scene_info": scene_info, "total_scenes": len(scene_info)},
                 )

            # --- Analyze Each Scene Frame ---
            logger.info("Analyzing frames from detected scenes...")
            # Use progressbar (or tqdm)
            widgets = [
                'Analyzing Scenes: ', progressbar.Percentage(),
                ' ', progressbar.Bar(marker=progressbar.RotatingMarker()),
                ' ', progressbar.ETA(),
                ' ', progressbar.FileTransferSpeed(),
            ]
            pbar = progressbar.ProgressBar(widgets=widgets, max_value=len(scene_info)).start()

            for i, scene in enumerate(scene_info):
                scene_id = scene.get("scene_id", f"scene_{i+1}") # Ensure scene_id exists

                # Skip if already processed
                if scene_id in processed_ids:
                    logger.debug(f"Skipping scene {scene_id} (already processed).")
                    pbar.update(i + 1)
                    continue

                logger.info(f"Processing scene {scene_id} ({scene['start_time']:.2f}s - {scene['end_time']:.2f}s)")
                scene_frame_analyses = []
                error_in_scene = False

                for frame_path in scene.get("frame_paths", []):
                    if not os.path.exists(frame_path):
                        logger.warning(f"Frame path not found: {frame_path}. Skipping.")
                        continue

                    try:
                        # Analyze the individual frame (similar to analyze_screenshot but simpler context)
                        frame_analysis_result = analyze_screenshot(
                            image_path=frame_path,
                            project_path=project_path, # Pass project path for consistency (though results might be overwritten)
                            software_list=software_list,
                            logo_db_path=logo_db_path,
                            ocr_lang=ocr_lang,
                            logo_threshold=logo_threshold,
                            context=(
                                f"This frame is from scene {scene_id} "
                                f"(time: {scene['start_time']:.2f}s - {scene['end_time']:.2f}s) of a video."
                            )
                        )
                        # Add frame path for reference within the scene analysis
                        frame_analysis_result["frame_path"] = frame_path
                        scene_frame_analyses.append(frame_analysis_result)

                        if "error" in frame_analysis_result:
                            error_in_scene = True # Mark if any frame analysis failed

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
                scene_analysis_result = {
                    "scene_id": scene_id,
                    "start_time": scene["start_time"],
                    "end_time": scene["end_time"],
                    "duration": scene.get("duration", scene["end_time"] - scene["start_time"]),
                    "frame_analyses": scene_frame_analyses,
                    "analysis_errors": error_in_scene, # Flag if errors occurred
                }

                # Add to overall results and save progress
                analyzed_results.append(scene_analysis_result)
                save_analyzed_segments(results_dir, analyzed_results) # Save after each scene

                # Save scene-specific checkpoint (optional but good for recovery)
                save_checkpoint(
                    project_path, f"SCENE_{scene_id}_ANALYZED", {"scene_id": scene_id, "analysis": scene_analysis_result}
                )
                pbar.update(i + 1)

            pbar.finish()
            logger.info("Scene detection and analysis complete.")

        else:
            # --- Segment-based Mode ---
            logger.info("Initiating segment-based analysis using ContextualFrameAnalyzer...")
            if not segments: # Should have been caught earlier, but double-check
                 raise ValueError("Segments list is empty for segment-based analysis.")

            # Initialize the contextual frame analyzer
            frame_analyzer = ContextualFrameAnalyzer(
                video_path=input_video,
                transcript_segments=segments, # Pass the full segments list here
                project_path=project_path, # Analyzer will create screenshots subdir
                software_list=software_list,
                logo_db_path=logo_db_path,
                ocr_lang=ocr_lang,
                logo_threshold=logo_threshold,
                quality_threshold=quality_threshold,
                save_format=save_format,
                compression_quality=compression_quality,
                # register=register # Pass register if ContextualFrameAnalyzer uses it
            )

            total_segments = len(segments)
            logger.info(f"Processing {total_segments} transcript segments...")

            # Use progressbar (or tqdm)
            widgets = [
                'Analyzing Segments: ', progressbar.Percentage(),
                ' ', progressbar.Bar(marker=progressbar.RotatingMarker()),
                ' ', progressbar.ETA(),
                ' ', progressbar.FileTransferSpeed(),
            ]
            pbar = progressbar.ProgressBar(widgets=widgets, max_value=total_segments).start()

            for i, segment in enumerate(segments):
                # Ensure segment has required keys for ContextualFrameAnalyzer
                segment_id = segment.get("segment_id")
                if segment_id is None:
                    logger.warning(f"Segment at index {i} is missing 'segment_id'. Assigning temporary ID: temp_{i+1}")
                    segment_id = f"temp_{i+1}"
                    segment["segment_id"] = segment_id # Add it for analyzer use

                # Skip if already processed
                if segment_id in processed_ids:
                    logger.debug(f"Skipping segment {segment_id} (already processed).")
                    pbar.update(i + 1)
                    continue

                logger.info(f"Processing segment {segment_id} ({segment.get('start_time', 'N/A'):.2f}s - {segment.get('end_time', 'N/A'):.2f}s)")

                try:
                    # Perform analysis using the dedicated analyzer class
                    analysis_result = frame_analyzer.analyze_segment(segment)

                    # Add to overall results and save progress
                    analyzed_results.append(analysis_result)
                    save_analyzed_segments(results_dir, analyzed_results) # Save after each segment

                    # Save segment-specific checkpoint
                    screenshot_paths = analysis_result.get("visual_summary", {}).get("screenshot_paths", [])
                    save_checkpoint(
                        project_path,
                        f"SEGMENT_{segment_id}_ANALYZED",
                        {"segment_id": segment_id, "analysis_summary": analysis_result.get("visual_summary"), "screenshots": screenshot_paths},
                    )
                    logger.debug(f"Completed segment {segment_id}. Screenshots: {len(screenshot_paths)}")

                except Exception as seg_err:
                    logger.error(f"Error processing segment {segment_id}: {seg_err}", exc_info=True)
                    # Add a placeholder error result?
                    analyzed_results.append({
                        "segment_id": segment_id,
                        "error": f"Failed to analyze segment: {seg_err}",
                        "visual_summary": {},
                        "frame_analyses": []
                    })
                    save_analyzed_segments(results_dir, analyzed_results) # Save progress even on error
                    # Continue to the next segment

                pbar.update(i + 1)

            pbar.finish()
            logger.info("Segment-based video analysis complete.")


        # --- Final Checkpoint ---
        logger.info("Saving final visual analysis checkpoint...")
        final_checkpoint_data = {
            "total_items_processed": len(analyzed_results),
            "results_path": os.path.join(results_dir, "visual_analysis_results.json"),
            "mode": "scene_detection" if extract_scenes else "segment_based",
        }
        if extract_scenes:
            final_checkpoint_data["scenes_dir"] = os.path.join(project_path, "scene_frames")
        else:
            final_checkpoint_data["screenshots_dir"] = os.path.join(project_path, "screenshots")

        save_checkpoint(
            project_path,
            CHECKPOINTS["VISUAL_ANALYSIS_COMPLETE"],
            final_checkpoint_data,
        )

        # Consider saving a final PROCESS_COMPLETE checkpoint here if this is the last step
        # save_checkpoint(project_path, CHECKPOINTS["PROCESS_COMPLETE"], {"status": "Success", "final_visual_results": final_checkpoint_data["results_path"]})


        return analyzed_results

    except FileNotFoundError as fnf_err:
        logger.error(f"{fnf_err}")
        raise # Re-raise specific error
    except ValueError as val_err:
        logger.error(f"Configuration error: {val_err}")
        raise # Re-raise specific error
    except Exception as e:
        error_msg = f"Critical error during visual analysis: {e}"
        logger.error(error_msg, exc_info=True)
        # Save whatever progress was made before raising
        save_analyzed_segments(results_dir, analyzed_results)
        raise RuntimeError(error_msg) from e
    finally:
        # --- Cleanup ---
        if frame_analyzer:
            logger.debug("Closing ContextualFrameAnalyzer resources.")
            frame_analyzer.close()

