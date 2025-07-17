#!/usr/bin/env python3
"""Visual analysis functionalities, including logo and OCR based software detection."""
import json
import logging
import os

import cv2
import numpy as np
import progressbar
from moviepy.editor import VideoFileClip
from PIL import Image, UnidentifiedImageError

from ..api.gemini import analyze_with_gemini
from ..constants import CHECKPOINTS
from ..processing.ocr.ocr_detection import detect_software_names
from ..processing.video.scene_detection import extract_scenes_from_video
from ..project import save_checkpoint
from ..prompt_templates import get_analysis_prompt
from .frame_analysis import ContextualFrameAnalyzer

logger = logging.getLogger(__name__)


def load_analyzed_segments(segments_dir):
    """Load any previously analyzed segments."""
    analysis_file = os.path.join(segments_dir, "analyzed_segments.json")
    if os.path.exists(analysis_file):
        with open(analysis_file, "r") as f:
            return json.load(f)
    return []


def save_analyzed_segments(segments_dir, analyzed_segments):
    """Save the current state of analyzed segments."""
    analysis_file = os.path.join(segments_dir, "analyzed_segments.json")
    with open(analysis_file, "w") as f:
        json.dump(analyzed_segments, f, indent=2)





def analyze_screenshot(
    image_path,
    project_path,
    software_list=None,
    ocr_lang="eng",
    context=None,
):
    """Analyze a single screenshot for software applications using OCR and logo detection."""
    from video_topic_splitter.project import load_checkpoint

    if not load_checkpoint(project_path):
        save_checkpoint(
            project_path, CHECKPOINTS["PROJECT_CREATED"], {"project_path": project_path}
        )
    try:
        # Read the image
        frame = cv2.imread(image_path)
        if frame is None:
            logger.warning(f"cv2.imread failed to load image: {image_path}")
            raise ValueError(f"Could not read image file: {image_path}")

        # Create PIL Image for Gemini
        image = Image.open(image_path)

        # Analyze frame for software
        ocr_matches = detect_software_names(frame, software_list, ocr_lang)

        analysis = {
            "ocr_matches": ocr_matches,
        }

        # Build software detection context
        software_context = ""
        if software_list:
            software_context = (
                f"\n\nDetected software applications:\n"
                f"Software list: {', '.join(software_list)}\n"
            )

            if analysis["ocr_matches"]:
                software_context += "\nText detected: " + ", ".join(
                    f"{m['software']} ({m['detected_text']})"
                    for m in analysis["ocr_matches"]
                )

        # Generate prompt for screenshot analysis
        prompt = (
            f"Analyze this screenshot and describe what you see.{software_context}\n\n"
            "Focus on identifying software applications, user interfaces, and any notable visual elements. "
            "Be specific about what is visible in the image."
        )

        # Add user-provided context if available
        if context:
            prompt = f"{context}\n\n{prompt}"

        # Get Gemini analysis
        try:
            gemini_analysis = analyze_with_gemini(prompt, image)
        except ValueError as ve:
            gemini_analysis = f"Gemini analysis configuration error: {ve}"
        except Exception as e:
            gemini_analysis = f"Gemini analysis failed: {e}"

        # Prepare results
        results = {
            "gemini_analysis": gemini_analysis,
            "software_detections": (
                [{"source": "screenshot", **analysis}]
                if (analysis["ocr_matches"])
                else []
            ),
        }


        # Save results
        results_path = os.path.join(project_path, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Save checkpoints
        save_checkpoint(
            project_path, CHECKPOINTS["SCREENSHOT_ANALYZED"], {"results": results}
        )
        save_checkpoint(
            project_path, CHECKPOINTS["PROCESS_COMPLETE"], {"results": results}
        )

        return results

    except Exception as e:
        logger.error(f"Error analyzing screenshot: {str(e)}")
        return {
            "gemini_analysis": f"Analysis failed: {str(e)}",
            "software_detections": [],
        }


def split_and_analyze_video(
    input_video,
    segments,
    project_path,
    software_list=None,
    ocr_lang="eng",
    quality_threshold=0.5,
    save_format="jpg",
    compression_quality=85,
    extract_scenes=False,
    min_scene_len=1.0,
    frames_per_scene=1,
    register="it-workflow",
):
    """Split video into segments and analyze each segment with checkpoint support.

    When extract_scenes is True, uses PySceneDetect to detect scene changes instead
    of splitting based on transcript segments.

    Args:
        input_video: Path to input video file
        segments: List of transcript segments
        project_path: Path to project directory
        software_list: Optional list of software names to detect
        ocr_lang: Language for OCR detection
        quality_threshold: Threshold for frame quality assessment (0-1)
        save_format: Format to save screenshots (jpg/png)
        compression_quality: JPEG compression quality (1-100)
        extract_scenes: Whether to use scene detection instead of segment splitting
        min_scene_len: Minimum scene length in seconds when using scene detection
        frames_per_scene: Number of frames to extract per scene when using scene detection
        register: Analysis register (it-workflow, gen-ai, tech-support)

    Returns:
        List of analyzed segments with visual summaries
    """
    print(f"Analyzing video: {input_video}")
    print("Splitting video into segments and analyzing...")

    # Create output directories
    os.makedirs(project_path, exist_ok=True)
    segments_dir = os.path.join(project_path, "segments")
    os.makedirs(segments_dir, exist_ok=True)

    # Load any previously analyzed segments
    analyzed_segments = load_analyzed_segments(segments_dir)

    # Create a map of existing analyses by segment_id
    existing_analyses = {seg["segment_id"]: seg for seg in analyzed_segments}

    try:
        if extract_scenes:
            # Use scene detection instead of segment-based analysis
            print("Using scene detection to extract frames...")

            # Create scenes directory
            scenes_dir = os.path.join(project_path, "scenes")
            os.makedirs(scenes_dir, exist_ok=True)

            # Detect scenes and extract frames
            scene_info = extract_scenes_from_video(
                input_video,
                scenes_dir,
                min_scene_len,
                threshold=27,  # Default threshold for content detector
                num_frames_per_scene=frames_per_scene,
                jpg_quality=compression_quality,
            )

            print(f"Detected {len(scene_info)} scenes")

            # Save scene detection checkpoint
            save_checkpoint(
                project_path,
                CHECKPOINTS["SCENES_DETECTED"],
                {
                    "scene_info": scene_info,
                    "total_scenes": len(scene_info),
                },
            )

            # Analyze each scene frame
            print("Analyzing scene frames...")

            # Create a PIL image for each frame for Gemini analysis
            for scene in progressbar.progressbar(scene_info):
                scene_id = scene["scene_id"]

                # Skip if this scene has already been fully processed
                if scene_id in existing_analyses:
                    print(f"\nSkipping scene {scene_id} (already processed)")
                    continue

                try:
                    # Process each frame in the scene
                    frame_analyses = []

                    for frame_path in scene["frame_paths"]:
                        # Read the image
                        frame = cv2.imread(frame_path)
                        if frame is None:
                            logger.warning(
                                f"cv2.imread failed to load image: {frame_path}"
                            )
                            continue

                        # Create PIL Image for Gemini
                        try:
                            image = Image.open(frame_path)
                        except UnidentifiedImageError:
                            logger.warning(
                                f"Failed to open image with PIL: {frame_path}"
                            )
                            continue

                        # Analyze frame for software
                        ocr_matches = detect_software_names(
                            frame, software_list, ocr_lang
                        )

                        software_analysis = {
                            "ocr_matches": ocr_matches,
                        }

                        # Build context for Gemini analysis
                        software_context = ""
                        if software_list:
                            if ocr_matches:
                                software_context += "\nText detected: " + ", ".join(
                                    f"{m['software']} ({m['detected_text']})"
                                    for m in ocr_matches
                                )

                        # Generate prompt for frame analysis
                        prompt = (
                            f"Analyze this frame from scene {scene_id} "
                            f"(time: {scene['start_time']:.2f}s - {scene['end_time']:.2f}s)."
                            f"{software_context}\n\n"
                            "Focus on identifying software applications, user interfaces, "
                            "and any notable visual elements. "
                            "Be specific about what is visible in the image."
                        )

                        # Get Gemini analysis
                        try:
                            gemini_analysis = analyze_with_gemini(prompt, image)
                        except Exception as e:
                            gemini_analysis = f"Analysis failed: {str(e)}"

                        # Add to frame analyses
                        frame_analyses.append(
                            {
                                "frame_path": frame_path,
                                "software_analysis": software_analysis,
                                "gemini_analysis": gemini_analysis,
                            }
                        )

                    # Create scene analysis result
                    analysis_result = {
                        "scene_id": scene_id,
                        "start_time": scene["start_time"],
                        "end_time": scene["end_time"],
                        "duration": scene["duration"],
                        "frame_paths": scene["frame_paths"],
                        "frame_analyses": frame_analyses,
                    }

                    # Save checkpoint for this scene
                    scene_checkpoint = {
                        "scene_id": scene_id,
                        "analysis": analysis_result,
                    }
                    save_checkpoint(
                        project_path, f"SCENE_{scene_id}_ANALYZED", scene_checkpoint
                    )

                    # Add to our results and save immediately
                    analyzed_segments.append(analysis_result)
                    save_analyzed_segments(segments_dir, analyzed_segments)

                    print(f"\nCompleted scene {scene_id}/{len(scene_info)}")

                except Exception as e:
                    print(f"\nError processing scene {scene_id}: {str(e)}")
                    # Save progress even if this scene failed
                    save_analyzed_segments(segments_dir, analyzed_segments)
                    continue

            print("\nScene detection and analysis complete.")

        else:
            # Use traditional segment-based analysis
            # Initialize the contextual frame analyzer with screenshot support
            frame_analyzer = ContextualFrameAnalyzer(
                video_path=input_video,
                transcript_segments=segments,
                project_path=project_path,
                software_list=software_list,
                ocr_lang=ocr_lang,
                quality_threshold=quality_threshold,
                save_format=save_format,
                compression_quality=compression_quality,
            )

            total_segments = len(segments)
            print(f"Processing {total_segments} segments...")

            for i, segment in enumerate(progressbar.progressbar(segments)):
                segment_id = i + 1
                segment_dir = os.path.join(segments_dir, f"segment_{segment_id}")
                os.makedirs(segment_dir, exist_ok=True)

                # Skip if this segment has already been fully processed
                if segment_id in existing_analyses:
                    print(f"\nSkipping segment {segment_id} (already processed)")
                    continue

                try:
                    # Analyze the segment with contextual frame analysis
                    analysis_result = frame_analyzer.analyze_segment(segment)

                    # Extract visual summary information
                    visual_summary = analysis_result.get("visual_summary", {})
                    screenshot_paths = visual_summary.get("screenshot_paths", [])

                    # Save checkpoint for this segment
                    segment_checkpoint = {
                        "segment_id": segment_id,
                        "analysis": analysis_result,
                        "screenshots": screenshot_paths,
                    }
                    save_checkpoint(
                        project_path,
                        f"SEGMENT_{segment_id}_ANALYZED",
                        segment_checkpoint,
                    )

                    # Add to our results and save immediately
                    analyzed_segments.append(analysis_result)
                    save_analyzed_segments(segments_dir, analyzed_segments)

                    print(f"\nCompleted segment {segment_id}/{total_segments}")
                    if screenshot_paths:
                        print(f"Saved {len(screenshot_paths)} screenshots")

                except Exception as e:
                    print(f"\nError processing segment {segment_id}: {str(e)}")
                    # Save progress even if this segment failed
                    save_analyzed_segments(segments_dir, analyzed_segments)
                    continue

            print("\nVideo splitting and analysis complete.")

            # Clean up frame analyzer resources
            frame_analyzer.close()

        # Save final checkpoint with visual analysis results
        if extract_scenes:
            save_checkpoint(
                project_path,
                CHECKPOINTS["VISUAL_ANALYSIS_COMPLETE"],
                {
                    "segments": analyzed_segments,
                    "total_segments": len(analyzed_segments),
                    "scenes_dir": os.path.join(project_path, "scenes"),
                },
            )
        else:
            save_checkpoint(
                project_path,
                CHECKPOINTS["VISUAL_ANALYSIS_COMPLETE"],
                {
                    "segments": analyzed_segments,
                    "total_segments": total_segments,
                    "screenshots_dir": os.path.join(project_path, "screenshots"),
                },
            )

        return analyzed_segments

    except Exception as e:
        error_msg = f"Error during video splitting and analysis: {str(e)}"
        print(f"\n{error_msg}")
        logger.error(f"Detailed error during video loading or processing: {str(e)}")
        # Save whatever progress we made
        save_analyzed_segments(segments_dir, analyzed_segments)
        raise RuntimeError(error_msg)
    finally:
        # Clean up frame analyzer resources
        if "frame_analyzer" in locals():
            frame_analyzer.close()
