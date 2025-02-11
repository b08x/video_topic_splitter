#!/usr/bin/env python3
"""Video analysis and segmentation functionality."""


import json
import logging
import os

import cv2
import google.generativeai as genai
import numpy as np
import progressbar
from dotenv import load_dotenv
from moviepy.editor import VideoFileClip
from PIL import Image, UnidentifiedImageError

from .constants import CHECKPOINTS
from .logo_detection import detect_software_logos
from .ocr_detection import detect_software_names
from .project import save_checkpoint
from .prompt_templates import get_analysis_prompt
from .prosodic_analysis import ProsodyAnalyzer
from .thumbnail_utils import ThumbnailManager

logger = logging.getLogger(__name__)
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Configure paths
LOGO_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "logos")


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
    logo_db_path=None,
    ocr_lang="eng",
    logo_threshold=0.8,
    context=None,
):
    """Analyze a single screenshot for software applications using OCR and logo detection.

    Args:
        image_path: Path to the image file
        project_path: Path to save results
        software_list: Optional list of software names to detect
        logo_db_path: Optional path to logo database
        ocr_lang: Language for OCR detection
        logo_threshold: Confidence threshold for logo detection

    Returns:
        dict: Analysis results including Gemini analysis and software detections
    """
    # Initialize project with PROJECT_CREATED checkpoint if not already initialized
    from .project import load_checkpoint

    if not load_checkpoint(project_path):
        save_checkpoint(
            project_path, CHECKPOINTS["PROJECT_CREATED"], {"project_path": project_path}
        )
    try:
        # Read the image
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image file: {image_path}")

        # Create PIL Image for Gemini
        image = Image.open(image_path)

        # Analyze frame for software
        analysis = analyze_frame_for_software(
            frame, software_list, logo_db_path, ocr_lang, logo_threshold
        )

        # Get Gemini analysis if API key is configured
        gemini_analysis = ""
        if os.getenv("GEMINI_API_KEY"):
            # Initialize Gemini model
            model = genai.GenerativeModel("gemini-2.0-flash-exp")

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

            # Generate content
            response = model.generate_content([prompt, image])
            gemini_analysis = response.text

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

        # Save checkpoints in sequence
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


def analyze_frame_for_software(
    frame, software_list=None, logo_db_path=None, ocr_lang="eng", logo_threshold=0.8
):
    """Analyze a single frame for software applications using OCR and logo detection."""
    results = {"ocr_matches": [], "logo_matches": []}

    if software_list:
        # Perform OCR detection
        ocr_matches = detect_software_names(frame, software_list, lang=ocr_lang)
        if ocr_matches:
            results["ocr_matches"] = ocr_matches

        # Perform logo detection
        logo_matches = detect_software_logos(
            frame, software_list, logo_db_path or LOGO_DB_PATH, threshold=logo_threshold
        )
        if logo_matches:
            results["logo_matches"] = logo_matches

    return results


def analyze_thumbnails(
    thumbnails,
    software_list=None,
    logo_db_path=None,
    ocr_lang="eng",
    logo_threshold=0.8,
    min_confidence=0.7,
):
    """Analyze thumbnails for software applications."""
    results = []

    for thumbnail_info in thumbnails:
        try:
            # Load thumbnail
            frame = cv2.imread(thumbnail_info["path"])
            if frame is None:
                continue

            # Analyze frame
            analysis = analyze_frame_for_software(
                frame, software_list, logo_db_path, ocr_lang, logo_threshold
            )

            # Calculate confidence score
            confidence = 0.0
            if analysis["ocr_matches"]:
                confidence = max(
                    match["confidence"] for match in analysis["ocr_matches"]
                )
            if analysis["logo_matches"]:
                logo_confidence = max(
                    match["confidence"] for match in analysis["logo_matches"]
                )
                confidence = max(confidence, logo_confidence)

            if confidence >= min_confidence:
                results.append(
                    {
                        "thumbnail": thumbnail_info,
                        "analysis": analysis,
                        "confidence": confidence,
                    }
                )

        except Exception as e:
            logger.error(
                f"Error analyzing thumbnail {thumbnail_info['path']}: {str(e)}"
            )
            continue

    return results


def analyze_segment_with_gemini(
    segment_path,
    transcript,
    software_list=None,
    logo_db_path=None,
    ocr_lang="eng",
    logo_threshold=0.8,
    min_thumbnail_confidence=0.7,
    register="gen-ai",
    include_prosody=True,
):
    """Analyze a video segment using Google's Gemini model and software detection."""
    print(f"Analyzing segment: {segment_path}")

    # Check if Gemini API key is configured
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY environment variable is not set")

    try:
        # Load the video segment
        video = VideoFileClip(segment_path)

        # Extract prosodic features if requested
        prosodic_features = None
        if include_prosody:
            try:
                prosody_analyzer = ProsodyAnalyzer()
                prosodic_features = prosody_analyzer.extract_prosodic_features(
                    segment_path
                )
            except Exception as e:
                logger.error(f"Error extracting prosodic features: {str(e)}")

        # Initialize ThumbnailManager
        thumbnail_manager = ThumbnailManager(os.path.dirname(segment_path))

        # Generate thumbnails
        thumbnails = thumbnail_manager.generate_thumbnails(
            segment_path, interval=5, max_thumbnails=5
        )

        # Analyze thumbnails
        thumbnail_results = analyze_thumbnails(
            thumbnails,
            software_list,
            logo_db_path,
            ocr_lang,
            logo_threshold,
            min_thumbnail_confidence,
        )

        # If thumbnail analysis confidence is low, analyze additional frames
        software_detections = []
        if (
            not thumbnail_results
            or max(r["confidence"] for r in thumbnail_results)
            < min_thumbnail_confidence
        ):
            duration = video.duration
            frame_times = [0, duration / 2, duration - 0.1]

            for time in frame_times:
                frame = video.get_frame(time)
                frame_results = analyze_frame_for_software(
                    frame, software_list, logo_db_path, ocr_lang, logo_threshold
                )
                if frame_results["ocr_matches"] or frame_results["logo_matches"]:
                    software_detections.append(
                        {"time": time, "source": "frame", **frame_results}
                    )

        # Add thumbnail results to software detections
        for result in thumbnail_results:
            software_detections.append(
                {
                    "time": result["thumbnail"].get("time", 0),
                    "source": "thumbnail",
                    "confidence": result["confidence"],
                    **result["analysis"],
                }
            )

        # Get first frame for Gemini analysis
        first_frame = video.get_frame(0)
        image = Image.fromarray(first_frame)
        video.close()

        # Build software detection context
        software_context = ""
        if software_list:
            software_context = (
                f"\n\nDetected software applications:\n"
                f"Software list: {', '.join(software_list)}\n"
            )

            if software_detections:
                for detection in software_detections:
                    software_context += f"\nAt {detection['time']:.2f}s:"
                    if detection["ocr_matches"]:
                        software_context += "\nText detected: " + ", ".join(
                            f"{m['software']} ({m['detected_text']})"
                            for m in detection["ocr_matches"]
                        )
                    if detection["logo_matches"]:
                        software_context += "\nLogos detected: " + ", ".join(
                            f"{m['software']} (confidence: {m['confidence']:.2f})"
                            for m in detection["logo_matches"]
                        )

        # Get register-specific analysis prompt
        prompt = get_analysis_prompt(register, software_context, transcript)

        # Initialize Gemini model
        model = genai.GenerativeModel("gemini-2.0-flash-exp")

        # Generate content
        response = model.generate_content([prompt, image])

        result = {
            "gemini_analysis": response.text,
            "software_detections": software_detections,
        }

        if prosodic_features:
            result["prosodic_features"] = prosodic_features

        return result
    except Exception as e:
        print(f"Error analyzing segment with Gemini: {str(e)}")
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
    register="gen-ai",
    include_prosody=True,
):
    """Split video into segments and analyze each segment with checkpoint support."""
    print("Splitting video into segments and analyzing...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize prosody analyzer if needed
    prosody_analyzer = None
    if include_prosody:
        try:
            prosody_analyzer = ProsodyAnalyzer()
        except Exception as e:
            logger.error(f"Failed to initialize prosody analyzer: {str(e)}")
            include_prosody = False

    # Load any previously analyzed segments
    analyzed_segments = load_analyzed_segments(output_dir)

    # Create a map of existing analyses by segment_id
    existing_analyses = {seg["segment_id"]: seg for seg in analyzed_segments}

    try:
        video = VideoFileClip(input_video)
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
                    start_time = segment["start_time"]
                    end_time = segment["end_time"]
                    segment_clip = video.subclip(start_time, end_time)
                    segment_clip.write_videofile(
                        output_path,
                        codec="libx264",
                        audio_codec="aac",
                        verbose=False,
                        logger=None,
                    )

                # Analyze the segment with software detection and prosody
                analysis_results = analyze_segment_with_gemini(
                    output_path,
                    segment["transcript"],
                    software_list,
                    logo_db_path,
                    ocr_lang,
                    logo_threshold,
                    min_thumbnail_confidence,
                    register,
                    include_prosody,
                )

                # Process prosodic features if available
                if include_prosody and prosody_analyzer:
                    try:
                        aligned_features = (
                            prosody_analyzer.align_features_with_transcript(
                                analysis_results.get("prosodic_features", {}), [segment]
                            )
                        )
                        if aligned_features:
                            analysis_results["aligned_prosodic_features"] = (
                                aligned_features[0]
                            )
                    except Exception as e:
                        logger.error(f"Error aligning prosodic features: {str(e)}")

                # Create the analysis result with software and prosodic information
                analysis_result = {
                    "segment_id": segment_id,
                    "start_time": segment["start_time"],
                    "end_time": segment["end_time"],
                    "transcript": segment["transcript"],
                    "topic": segment["dominant_topic"],
                    "keywords": segment["top_keywords"],
                    "gemini_analysis": analysis_results["gemini_analysis"],
                    "software_detected": bool(software_list),
                    "software_detections": analysis_results["software_detections"],
                }

                # Add prosodic features if available
                if "aligned_prosodic_features" in analysis_results:
                    analysis_result["prosodic_features"] = analysis_results[
                        "aligned_prosodic_features"
                    ]

                # Add to our results and save immediately
                analyzed_segments.append(analysis_result)
                save_analyzed_segments(output_dir, analyzed_segments)

                print(f"\nCompleted segment {segment_id}/{total_segments}")

            except Exception as e:
                print(f"\nError processing segment {segment_id}: {str(e)}")
                # Save progress even if this segment failed
                save_analyzed_segments(output_dir, analyzed_segments)
                continue

        video.close()
        print("\nVideo splitting and analysis complete.")
        return analyzed_segments

    except Exception as e:
        print(f"\nError during video splitting and analysis: {str(e)}")
        # Save whatever progress we made
        save_analyzed_segments(output_dir, analyzed_segments)
        raise
    finally:
        # Make sure we always close the video file
        if "video" in locals():
            video.close()
