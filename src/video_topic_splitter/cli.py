#!/usr/bin/env python3
"""Command-line interface for video scene splitter."""

import argparse
import logging
import os
import sys
import json
from typing import Optional, Tuple

from dotenv import load_dotenv

# Updated imports
from video_topic_splitter.constants import CHECKPOINTS, DEFAULT_SOFTWARE_LIST
from video_topic_splitter.core import process_video
from video_topic_splitter.project import create_project_folder, load_checkpoint
from video_topic_splitter.utils.youtube import is_youtube_url
from video_topic_splitter.pipeline.visual_topic_pipeline import run_visual_topic_pipeline

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
        description="Process video to detect scenes, analyze content, and split."
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
    # --- Visual Analysis Args ---
    parser.add_argument(
        "--software-list", type=str,
        help="Path to a text file containing software names for OCR detection (one per line)"
    )
    parser.add_argument(
        "--ocr-lang", default="eng", help="Language for OCR detection (default: eng)"
    )
    parser.add_argument(
        "--frames-per-scene", type=int, default=1,
        help="Number of frames to extract/analyze per scene (default: 1)"
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
        help="Analysis register for Gemini context: it-workflow, gen-ai, tech-support, or educational"
    )
    
    # --- Visual Topic Analysis Args ---
    parser.add_argument(
        "--visual-topic", action="store_true",
        help="Enable enhanced visual topic analysis that correlates transcript with visual elements"
    )
    parser.add_argument(
        "--visual-similarity-threshold", type=float, default=0.6,
        help="Threshold for visual similarity detection (0.0-1.0, default: 0.6)"
    )
    parser.add_argument(
        "--force-reanalysis", action="store_true",
        help="Force reanalysis of already processed data"
    )
    parser.add_argument(
        "--extract-insights", action="store_true",
        help="Extract key insights from the analysis results"
    )
    # Removed: --topics, --transcribe-only, --logo-db, --logo-threshold, thumbnail args

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
    software_list = DEFAULT_SOFTWARE_LIST  # Use default list as a starting point
    if args.software_list:
        if not os.path.exists(args.software_list):
            logger.error(f"Software list file not found: {args.software_list}")
            sys.exit(1)
        try:
            with open(args.software_list, "r", encoding='utf-8') as f:
                software_list = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(software_list)} software names to detect via OCR.")
        except Exception as e:
            logger.error(f"Failed to read software list file: {e}", exc_info=True)
            sys.exit(1)
    else:
        logger.info(f"Using default software list with {len(software_list)} entries for OCR detection.")
            
    # Check for GEMINI_API_KEY if visual topic analysis is enabled
    if args.visual_topic:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            logger.error("GEMINI_API_KEY environment variable not found. Please set it in your .env file.")
            sys.exit(1)

    try:
        # Check if we should run visual topic analysis
        if args.visual_topic:
            logger.info("Running enhanced visual topic analysis pipeline...")
            try:
                results = run_visual_topic_pipeline(
                    input_video=args.input,
                    project_path=project_path,
                    gemini_api_key=os.getenv("GEMINI_API_KEY"),
                    software_list=software_list,
                    ocr_lang=args.ocr_lang,
                    frames_per_scene=args.frames_per_scene,
                    frame_format=args.frame_format,
                    compression_quality=args.frame_quality,
                    register=args.register,
                    force_reanalysis=args.force_reanalysis,
                    visual_similarity_threshold=args.visual_similarity_threshold
                )
                
                # If insights extraction is requested
                if args.extract_insights and results:
                    from video_topic_splitter.pipeline.visual_topic_pipeline import extract_key_insights
                    insights = extract_key_insights(results)
                    insights_path = os.path.join(project_path, "key_insights.json")
                    
                    try:
                        with open(insights_path, 'w', encoding='utf-8') as f:
                            json.dump(insights, f, indent=2, ensure_ascii=False)
                        logger.info(f"Key insights saved to: {insights_path}")
                        
                        # Print a summary to console
                        print("\n===== KEY INSIGHTS =====")
                        print(f"Major Topics: {len(insights['major_topics'])}")
                        for topic in insights['major_topics'][:3]:  # Show top 3
                            print(f"- {topic['name']}: {', '.join(topic['keywords'][:3])}")
                        
                        print(f"\nKey Visual Elements: {len(insights['key_visual_elements'])}")
                        for element in insights['key_visual_elements'][:3]:  # Show top 3
                            print(f"- {element['name']}: {', '.join(element['keywords'][:3])}")
                        
                        print(f"\nSignificant Moments: {len(insights['significant_moments'])}")
                        for moment in insights['significant_moments'][:3]:  # Show top 3
                            print(f"- {moment['time_range']}: {moment['text_topic']} / {moment['visual_topic']}")
                        
                        if insights['software_usage']:
                            print(f"\nSoftware Usage: {len(insights['software_usage'])}")
                            for sw in insights['software_usage'][:3]:  # Show top 3
                                print(f"- {sw['name']}: {len(sw['time_ranges'])} occurrences")
                        
                        print("=======================\n")
                        
                    except Exception as e:
                        logger.error(f"Failed to save or display insights: {e}")
                
                logger.info("Visual topic analysis pipeline completed successfully.")
                
            except Exception as e:
                logger.error(f"Visual topic analysis pipeline failed: {e}", exc_info=True)
                sys.exit(1)
        else:
            # Check if already complete
            checkpoint = load_checkpoint(project_path)
            if checkpoint and checkpoint["stage"] == CHECKPOINTS["PROCESS_COMPLETE"]:
                logger.info("Process already completed previously. Loading final results.")
                final_results_path = checkpoint["data"].get("final_results_path")
                if final_results_path and os.path.exists(final_results_path):
                     with open(final_results_path, 'r', encoding='utf-8') as f:
                         results = json.load(f)
                     logger.info(f"Results loaded from {final_results_path}")
                     # Optionally print summary here
                else:
                     logger.warning("Process complete checkpoint found, but results file missing. Re-running.")
                     # Call process_video to potentially regenerate results
                     results = process_video(
                        input_path=args.input, # Use original input here
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
                     )

            else:
                # Run the main processing pipeline
                results = process_video(
                    input_path=args.input, # Use original input here
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
                )

        # --- Output Summary ---
        logger.info("--- Processing Summary ---")
        logger.info(f"Project Folder: {project_path}")
        if results.get("error"):
            logger.error(f"Processing finished with error: {results['error']}")
        else:
            num_scenes = len(results.get("scene_boundaries", []))
            num_analyzed = len(results.get("scene_analysis", []))
            num_splits = len(results.get("split_video_paths", []))
            logger.info(f"Detected Scenes: {num_scenes}")
            logger.info(f"Analyzed Scenes: {num_analyzed}")
            logger.info(f"Split Video Files Created: {num_splits}")
            logger.info(f"Final results JSON saved in project folder.")
            if num_splits > 0:
                 logger.info(f"Split videos saved in: {os.path.join(project_path, 'split_videos')}")

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
