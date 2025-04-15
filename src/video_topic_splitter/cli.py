#!/usr/bin/env python3
"""Command-line interface for video scene splitter."""

import argparse
import logging
import os
import sys
import json
from typing import Optional, Tuple, List # Added List

from dotenv import load_dotenv

# Updated imports
from video_topic_splitter.constants import CHECKPOINTS, DEFAULT_SOFTWARE_LIST # Use updated CHECKPOINTS
from video_topic_splitter.core import process_video # Use the refactored process_video
from video_topic_splitter.project import create_project_folder, load_checkpoint
from video_topic_splitter.utils.youtube import is_youtube_url
# Removed import for visual_topic_pipeline as it's no longer called directly here
# from video_topic_splitter.pipeline.visual_topic_pipeline import run_visual_topic_pipeline

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_input(input_path: str) -> Tuple[Optional[str], bool]:
    """Validate input is either a valid file path or YouTube URL."""
    if is_youtube_url(input_path):
        return None, True

    if not os.path.exists(input_path):
        return f"Input file not found: {input_path}", False

    # Allow common video formats
    if not input_path.lower().endswith((".mp4", ".mkv", ".avi", ".mov", ".webm")):
        return f"Unsupported video file format. Please use common formats like .mp4, .mkv, etc.", False

    return None, False


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Process video to detect scenes, analyze content using multimodal topic modeling, and split."
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Path to input video file or YouTube URL"
    )
    parser.add_argument(
        "-o", "--output", default=os.getcwd(), help="Base output directory for project folders"
    )
    parser.add_argument(
        "--api", choices=["deepgram"], default="deepgram",
        help="Choose transcription API (currently only 'deepgram' supported)"
    )
    # --- Scene Detection Args ---
    parser.add_argument(
        "--scene-threshold", type=float, default=27.0,
        help="Threshold for PySceneDetect ContentDetector (default: 27.0)"
    )
    parser.add_argument(
        "--min-scene-len", type=float, default=1.0,
        help="Minimum scene length in seconds (default: 1.0)"
    )
    # --- Audio Processing Arg ---
    parser.add_argument(
        "--skip-unsilence", action="store_true", help="Skip silence removal processing"
    )
    # --- Visual Analysis & Topic Modeling Args (Now integrated) ---
    parser.add_argument(
        "--software-list", type=str,
        help="Path to a text file containing software names for OCR detection (one per line)"
    )
    parser.add_argument(
        "--ocr-lang", default="eng", help="Language for OCR detection (default: eng)"
    )
    parser.add_argument(
        "--frames-per-scene", type=int, default=3, # Default updated to 3 based on visual_analysis defaults
        help="Number of frames to extract/analyze per scene (default: 3)"
    )
    parser.add_argument(
        "--frame-format", choices=["jpg", "png"], default="jpg",
        help="Format for extracted frames (default: jpg)"
    )
    parser.add_argument(
        "--frame-quality", type=int, default=90,
        help="Quality for JPEG frames (1-100, default: 90)"
    )
    parser.add_argument(
        "--register", choices=["it-workflow", "gen-ai", "tech-support", "educational"],
        default="it-workflow",
        help="Analysis register for context: it-workflow, gen-ai, tech-support, or educational"
    )
    parser.add_argument(
        "--visual-similarity-threshold", type=float, default=0.6, # Kept for VisualTopicAnalyzer
        help="Threshold for visual similarity detection (0.0-1.0, default: 0.6)"
    )
    # --- General Args ---
    parser.add_argument(
        "--force-reanalysis", action="store_true",
        help="Force reanalysis of already processed data (by ignoring checkpoints)"
    )
    parser.add_argument(
        "--extract-insights", action="store_true",
        help="Extract key insights from the final visual topic analysis results"
    )
    # Removed: --visual-topic flag (now default behavior)

    args = parser.parse_args()

    # Load environment variables (.env file)
    load_dotenv()

    # Validate input
    error, is_youtube = validate_input(args.input)
    if error:
        logger.error(f"Input validation failed: {error}")
        sys.exit(1)

    # Create project folder
    try:
        project_path = create_project_folder(args.input, args.output)
        logger.info(f"Using project folder: {project_path}")
    except Exception as e:
        logger.error(f"Failed to create project folder: {e}", exc_info=True)
        sys.exit(1)

    # Load software list if provided, otherwise use DEFAULT_SOFTWARE_LIST
    software_list: Optional[List[str]] = DEFAULT_SOFTWARE_LIST  # Use default list
    if args.software_list:
        if not os.path.exists(args.software_list):
            logger.error(f"Software list file not found: {args.software_list}")
            sys.exit(1)
        try:
            with open(args.software_list, "r", encoding='utf-8') as f:
                # Filter out empty lines
                loaded_list = [line.strip() for line in f if line.strip()]
            if loaded_list: # Only override default if file contains names
                 software_list = loaded_list
                 logger.info(f"Loaded {len(software_list)} software names from {args.software_list}.")
            else:
                 logger.warning(f"Software list file {args.software_list} was empty. Using default list.")
                 software_list = DEFAULT_SOFTWARE_LIST # Explicitly reset to default
        except Exception as e:
            logger.error(f"Failed to read software list file: {e}", exc_info=True)
            sys.exit(1)
    else:
        logger.info(f"Using default software list with {len(software_list)} entries for OCR detection.")

    # Check required API Keys (Gemini is now always needed for the unified pipeline)
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY environment variable not found. Please set it in your .env file.")
        sys.exit(1)
    if not os.getenv("OPENROUTER_API_KEY"):
         logger.error("OPENROUTER_API_KEY environment variable not found. Please set it in your .env file.")
         sys.exit(1)
    if args.api == "deepgram" and not os.getenv("DG_API_KEY"):
         logger.error("DG_API_KEY environment variable not found (required for Deepgram). Please set it.")
         sys.exit(1)
    # Add checks for other APIs if supported in the future

    try:
        # Check if already complete using the updated checkpoint stage value
        checkpoint = load_checkpoint(project_path)
        process_complete_stage = CHECKPOINTS["PROCESS_COMPLETE"] # Use updated value

        # Always run process_video, but it will handle checkpoints internally
        # (Remove the check here unless force_reanalysis should bypass internal checks too)
        # if checkpoint and checkpoint["stage"] == process_complete_stage and not args.force_reanalysis:
        #     logger.info("Process already completed previously. Loading final results.")
        #     # ... (load existing results logic) ...
        # else:

        logger.info("Starting main processing pipeline (with integrated visual analysis)...")
        # Call the unified process_video function, passing all relevant arguments
        results = process_video(
            input_path=args.input,
            project_path=project_path,
            api=args.api,
            skip_unsilence=args.skip_unsilence,
            scene_threshold=args.scene_threshold,
            min_scene_len=args.min_scene_len,
            software_list=software_list,
            ocr_lang=args.ocr_lang,
            frames_per_scene=args.frames_per_scene,
            frame_format=args.frame_format,
            compression_quality=args.frame_quality,
            register=args.register,
            visual_similarity_threshold=args.visual_similarity_threshold # Pass this arg
        )

        # Handle insights extraction if requested
        if args.extract_insights and results and not results.get("error"):
            logger.info("Extracting key insights...")
            # Import locally or ensure it's moved/available
            try:
                from video_topic_splitter.analysis.visual_topic_modeling import extract_key_insights # Assuming moved here
                # Pass the main results dict; extraction function needs results structure
                insights = extract_key_insights(results.get("visual_topic_analysis", {})) # Adapt based on actual results structure
                insights_path = os.path.join(project_path, "key_insights.json")

                with open(insights_path, 'w', encoding='utf-8') as f:
                    json.dump(insights, f, indent=2, ensure_ascii=False)
                logger.info(f"Key insights saved to: {insights_path}")

                # Print a summary to console (optional, adapt as needed)
                print("\n===== KEY INSIGHTS =====")
                # ... (print summary logic - adapt based on insight structure) ...
                print("=======================\n")

            except ImportError:
                 logger.error("Could not import 'extract_key_insights'. Insights extraction skipped.")
            except Exception as e:
                logger.error(f"Failed to extract or save insights: {e}")

        # --- Output Summary ---
        logger.info("--- Processing Summary ---")
        logger.info(f"Project Folder: {project_path}")
        if results.get("error"):
            logger.error(f"Processing finished with error: {results['error']}")
        else:
            # Extract info from the visual_topic_analysis part of results
            analysis_results = results.get("visual_topic_analysis", {})
            num_segments = len(analysis_results.get("segments", []))
            num_splits = len(results.get("split_video_paths", []))

            logger.info(f"Identified Segments (Visual+Text): {num_segments}")
            logger.info(f"Split Video Files Created: {num_splits}")
            logger.info(f"Final results JSON saved in project folder.")
            if num_splits > 0:
                 split_dir = os.path.join(project_path, 'split_videos')
                 logger.info(f"Split videos saved in: {split_dir}")

    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user. Progress may have been saved via checkpoints.")
        logger.info("To resume, run the same command again.")
        sys.exit(1)
    except FileNotFoundError as e:
         logger.error(f"File not found error during processing: {e}", exc_info=True)
         sys.exit(1)
    except ValueError as e:
         logger.error(f"Configuration or value error during processing: {e}", exc_info=True)
         sys.exit(1)
    except RuntimeError as e:
         logger.error(f"Runtime error during processing: {e}", exc_info=True)
         sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        logger.info("Progress may have been saved via checkpoints. Try running the command again.")
        sys.exit(1)


if __name__ == "__main__":
    main()