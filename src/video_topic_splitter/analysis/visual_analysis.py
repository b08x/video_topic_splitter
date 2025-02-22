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

def detect_software_logos(frame, software_list=None, logo_db_path=None, threshold=0.8):
    """Analyze a frame for software logos using template matching."""
    results = []

    if not software_list or not logo_db_path or not os.path.exists(logo_db_path):
        return results

    for software in software_list:
        # Look for logo template files
        logo_path = os.path.join(logo_db_path, f"{software.lower()}.png")
        if not os.path.exists(logo_path):
            continue

        try:
            # Read and match template
            template = cv2.imread(logo_path)
            if template is None:
                continue

            # Convert both to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

            # Template matching
            result = cv2.matchTemplate(gray_frame, gray_template, cv2.TM_CCOEFF_NORMED)

            # Get matches above threshold
            locations = np.where(result >= threshold)
            for pt in zip(*locations[::-1]):  # Switch columns and rows
                results.append(
                    {
                        "software": software,
                        "confidence": float(result[pt[1]][pt[0]]),
                        "location": {"x": int(pt[0]), "y": int(pt[1])},
                    }
                )

        except Exception as e:
            logger.error(f"Error matching logo for {software}: {str(e)}")
            continue

    return results

# Configure paths
LOGO_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "logos")

def analyze_screenshot(
    image_path,
    project_path,
    software_list=None,
    logo_db_path=None,
    ocr_lang="eng",
    logo_threshold=0.8,
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
        logo_matches = detect_software_logos(
            frame, software_list, logo_db_path, logo_threshold
        )

        analysis = {
            "ocr_matches": ocr_matches,
            "logo_matches": logo_matches,
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
            if analysis["logo_matches"]:
                software_context += "\nLogos detected: " + ", ".join(
                    f"{m['software']} (confidence: {m['confidence']:.2f})"
                    for m in analysis["logo_matches"]
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
                if (analysis["ocr_matches"] or analysis["logo_matches"])
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
    output_dir,
    software_list=None,
    logo_db_path=None,
    ocr_lang="eng",
    logo_threshold=0.8,
    thumbnail_interval=5,
    max_thumbnails=5,
    min_thumbnail_confidence=0.7,
    register="it-workflow",
):
    """Split video into segments and analyze each segment with checkpoint support."""
    print(f"Analyzing video: {input_video}")
    print("Splitting video into segments and analyzing...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load any previously analyzed segments
    analyzed_segments = load_analyzed_segments(output_dir)

    # Create a map of existing analyses by segment_id
    existing_analyses = {seg["segment_id"]: seg for seg in analyzed_segments}

    try:
        # Initialize the contextual frame analyzer
        frame_analyzer = ContextualFrameAnalyzer(
            input_video,
            segments,
            software_list,
            logo_db_path,
            ocr_lang,
            logo_threshold,
        )

        total_segments = len(segments)
        print(f"Processing {total_segments} segments...")

        for i, segment in enumerate(progressbar.progressbar(segments)):
            segment_id = i + 1
            output_path = os.path.join(output_dir, f"segment_{segment_id}.mp4")

            # Skip if this segment has already been fully processed
            if segment_id in existing_analyses:
                print(f"\nSkipping segment {segment_id} (already processed)")
                continue

            try:
                # Extract and save the segment
                if not os.path.exists(output_path):
                    video = VideoFileClip(input_video)
                    start_time = segment["start_time"]
                    end_time = segment["end_time"]
                    segment_clip = video.subclip(start_time, end_time)
                    segment_clip.write_videofile(
                        output_path,
                        codec="libx264",
                        audio_codec="aac",
                        fps=video.fps or 24,
                        verbose=False,
                        logger=None,
                    )
                    video.close()

                # Analyze the segment with contextual frame analysis
                analysis_result = frame_analyzer.analyze_segment(segment)

                # Add to our results and save immediately
                analyzed_segments.append(analysis_result)
                save_analyzed_segments(output_dir, analyzed_segments)

                print(f"\nCompleted segment {segment_id}/{total_segments}")

            except Exception as e:
                print(f"\nError processing segment {segment_id}: {str(e)}")
                # Save progress even if this segment failed
                save_analyzed_segments(output_dir, analyzed_segments)
                continue

        print("\nVideo splitting and analysis complete.")
        return analyzed_segments

    except Exception as e:
        error_msg = f"Error during video splitting and analysis: {str(e)}"
        print(f"\n{error_msg}")
        logger.error(f"Detailed error during video loading or processing: {str(e)}")
        # Save whatever progress we made
        save_analyzed_segments(output_dir, analyzed_segments)
        raise RuntimeError(error_msg)
    finally:
        # Clean up frame analyzer resources
        if 'frame_analyzer' in locals():
            frame_analyzer.close()
