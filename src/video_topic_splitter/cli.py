# cli.py
"""
Command-line interface for the video topic splitter.

This module provides a CLI for processing videos or screenshots for topic-based
segmentation, scene analysis, and multimodal analysis. It handles argument
parsing, input validation, and orchestrates the core processing functions.

Example usage:
    # Process a local video file
    python -m video_topic_splitter.cli -i /path/to/video.mp4 -o /path/to/output

    # Process a YouTube video
    python -m video_topic_splitter.cli -i "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    # Analyze a single screenshot
    python -m video_topic_splitter.cli -i /path/to/screenshot.png --analyze-screenshot
"""

import argparse
import os
import sys
from typing import Optional, Tuple

from dotenv import load_dotenv

from .constants import CHECKPOINTS
from .core import process_video
from .project import create_project_folder, load_checkpoint
from .utils.youtube import is_youtube_url


def validate_input(
    input_path: str, transcript_path: Optional[str], analyze_screenshot: bool
) -> Tuple[Optional[str], bool]:
    """
    Validate that inputs are valid files or a YouTube URL.

    Checks for the existence of local files and validates file extensions for
    videos, transcripts, and images.

    Args:
        input_path: The path to the input video, image, or a YouTube URL.
        transcript_path: The optional path to a transcript file.
        analyze_screenshot: Flag indicating if the input is a screenshot.

    Returns:
        A tuple containing an error message string (or None if valid) and a
        boolean indicating if the input is a YouTube URL.
    """
    is_youtube = is_youtube_url(input_path)

    if not is_youtube and not os.path.exists(input_path):
        return f"Input file not found: {input_path}", False

    if transcript_path and not os.path.exists(transcript_path):
        return f"Transcript file not found: {transcript_path}", False

    if analyze_screenshot:
        if not input_path.lower().endswith((".png", ".jpg", ".jpeg")):
            return "Unsupported image format. Use .png, .jpg, or .jpeg.", False
    elif not is_youtube:
        if not input_path.lower().endswith((".mp4", ".mkv")):
            return "Unsupported video format. Use .mp4 or .mkv.", False

    if transcript_path:
        if not transcript_path.lower().endswith((".srt", ".vtt", ".json")):
            return "Unsupported transcript format. Use .srt, .vtt, or .json.", False

    return None, is_youtube


def main() -> None:
    """
    Main entry point for the CLI.

    Parses command-line arguments, validates inputs, creates a project
    structure, and initiates the video or screenshot processing pipeline.
    Handles checkpointing to resume progress and manages exceptions.
    """
    parser = argparse.ArgumentParser(
        description="Process video for topic-based segmentation and scene analysis."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to input video file (mp4, mkv) or a YouTube URL.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=os.getcwd(),
        help="Base output directory for project folders.",
    )
    parser.add_argument(
        "--transcript",
        help="Optional path to a transcript file (.srt, .vtt, .json) to skip transcription.",
    )
    parser.add_argument(
        "--topics", type=int, default=5, help="Number of topics for topic modeling."
    )
    parser.add_argument(
        "--register",
        choices=["it-workflow", "gen-ai", "tech-support"],
        default="it-workflow",
        help="Analysis register for tailoring Gemini's analysis.",
    )
    parser.add_argument(
        "--skip-unsilence", action="store_true", help="Skip silence removal processing."
    )
    parser.add_argument(
        "--transcribe-only",
        action="store_true",
        help="Only perform audio transcription; skips all other analysis.",
    )
    parser.add_argument(
        "--software-list",
        type=str,
        help="Path to a text file with software names to detect (one per line).",
    )
    parser.add_argument(
        "--ocr-lang", default="eng", help="Language for OCR detection (default: eng)."
    )
    parser.add_argument(
        "--frames-per-scene",
        type=int,
        default=1,
        help="Number of unique frames to extract per detected scene (default: 1).",
    )
    parser.add_argument(
        "--analyze-screenshot",
        action="store_true",
        help="Analyze a single screenshot instead of a video.",
    )
    parser.add_argument(
        "--screenshot-context",
        help="Optional context for screenshot analysis.",
    )
    parser.add_argument(
        "--progress-json",
        action="store_true",
        help="Output progress information as JSON for programmatic consumption.",
    )

    args = parser.parse_args()
    load_dotenv()

    error, is_youtube = validate_input(
        args.input, args.transcript, args.analyze_screenshot
    )
    if error:
        print(f"Error: {error}")
        sys.exit(1)

    try:
        project_path = create_project_folder(args.input, args.output)
        print(f"Project folder: {project_path}")
    except Exception as e:
        print(f"Error creating project folder: {str(e)}")
        sys.exit(1)

    try:
        checkpoint = load_checkpoint(project_path)
        if checkpoint and checkpoint["stage"] == CHECKPOINTS["PROCESS_COMPLETE"]:
            print("Loading results from previous complete run.")
            results = checkpoint["data"]["results"]
        elif args.analyze_screenshot:
            from .analysis.visual_analysis import analyze_screenshot
            results = analyze_screenshot(
                args.input,
                project_path,
                software_list=args.software_list,
                ocr_lang=args.ocr_lang,
                context=args.screenshot_context,
            )
        else:
            software_list = None
            if args.software_list:
                if not os.path.exists(args.software_list):
                    print(f"Error: Software list file not found: {args.software_list}")
                    sys.exit(1)
                with open(args.software_list, "r") as f:
                    software_list = [line.strip() for line in f if line.strip()]
                print(f"Loaded {len(software_list)} software applications to detect.")

            results = process_video(
                video_path=args.input,
                project_path=project_path,
                transcript_path=args.transcript,
                num_topics=args.topics,
                skip_unsilence=args.skip_unsilence,
                transcribe_only=args.transcribe_only,
                is_youtube_url=is_youtube,
                software_list=software_list,
                ocr_lang=args.ocr_lang,
                frames_per_scene=args.frames_per_scene,
                register=args.register,
                progress_json=args.progress_json,
            )

        print(f"\nProcessing complete. Project folder: {project_path}")
        print(f"Results saved in: {os.path.join(project_path, 'results.json')}")

        if not args.transcribe_only and not args.analyze_screenshot:
            print("\nTopics and Segments:")
            for topic in results.get("topics", []):
                print(f"  Topic {topic['topic_id'] + 1}: {', '.join(topic['words'])}")
            print(f"\nAnalyzed {len(results.get('analyzed_scenes', []))} scenes.")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Progress has been saved.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred during processing: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
