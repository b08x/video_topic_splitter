# processing/video/video_analysis.py
#!/usr/bin/env python3
"""Video analysis and segmentation functionality."""


import json
import logging
import os

import cv2
import numpy as np
import progressbar
from moviepy.editor import VideoFileClip
from PIL import Image, UnidentifiedImageError
from video_topic_splitter.analysis.visual_analysis import \
    detect_software_logos  # Corrected import path
from video_topic_splitter.api.gemini import \
    analyze_with_gemini  # Corrected import path
from video_topic_splitter.constants import CHECKPOINTS
from video_topic_splitter.processing.ocr.ocr_detection import \
    detect_software_names
from video_topic_splitter.project import save_checkpoint
from video_topic_splitter.prompt_templates import get_analysis_prompt
from video_topic_splitter.utils.thumbnail import ThumbnailManager

logger = logging.getLogger(__name__)


# Configure paths - LOGO_DB_PATH is now used in logo_detection.py directly.
LOGO_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "logos"
)  # Adjusted path


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


# analyze_screenshot moved to analysis/visual_analysis.py
# analyze_frame_for_software moved to analysis/visual_analysis.py
# analyze_thumbnails moved to analysis/visual_analysis.py
# analyze_segment_with_gemini moved to analysis/visual_analysis.py


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
    print("Splitting video into segments and analyzing...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

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

                # Analyze the segment with software detection - moved to analysis/visual_analysis.py and called from there.
                from video_topic_splitter.analysis.visual_analysis import \
                    analyze_segment_with_gemini  # Import here to avoid circular import.

                analysis_results = analyze_segment_with_gemini(  # Calling from analysis/visual_analysis.py
                    output_path,
                    segment["transcript"],
                    software_list,
                    logo_db_path,
                    ocr_lang,
                    logo_threshold,
                    min_thumbnail_confidence,
                    register,
                )

                # Create the analysis result with software information
                analysis_result = {
                    "segment_id": segment_id,
                    "start_time": segment["start_time"],
                    "end_time": segment["end_time"],
                    "transcript": segment["transcript"],
                    "topic": segment["dominant_topic"],
                    "keywords": segment["top_keywords"],
                    "gemini_analysis": analysis_results["gemini_analysis"],
                    "software_detected": bool(
                        software_list
                    ),  # Indicates if software detection was performed
                    "software_detections": analysis_results["software_detections"],
                }

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
