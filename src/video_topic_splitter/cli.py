# cli.py
"""Command-line interface for video topic splitter."""

import argparse
import os
import sys
from typing import Optional, Tuple

from dotenv import load_dotenv
from video_topic_splitter.constants import CHECKPOINTS
from video_topic_splitter.core import process_video
from video_topic_splitter.project import create_project_folder, load_checkpoint
from video_topic_splitter.utils.youtube import is_youtube_url


def validate_input(
    input_path: str, analyze_screenshot: bool = False
) -> Tuple[Optional[str], bool]:
    """Validate input is either a valid file path or YouTube URL.

    Args:
        input_path: Path to input file or YouTube URL
        analyze_screenshot: Whether we're analyzing a screenshot (allows image files)

    Returns:
        Tuple[Optional[str], bool]: (error message if any, is_youtube_url)
    """
    # Check if input is a YouTube URL
    if is_youtube_url(input_path):
        return None, True

    # Check if input is a valid file
    if not os.path.exists(input_path):
        return f"Input file not found: {input_path}", False

    if analyze_screenshot:
        if not input_path.lower().endswith((".png", ".jpg", ".jpeg")):
            return (
                f"Unsupported image format. Supported formats: .png, .jpg, .jpeg",
                False,
            )
    else:
        if not input_path.lower().endswith((".mp4", ".mkv", ".json")):
            return (
                f"Unsupported file format. Supported formats: .mp4, .mkv, .json",
                False,
            )

    return None, False


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Process video/transcript for topic-based segmentation or analyze screenshots"
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to input video file, YouTube URL, transcript JSON, or image file (with --analyze-screenshot)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=os.getcwd(),
        help="Base output directory for project folders",
    )
    parser.add_argument(
        "--api",
        choices=["deepgram", "groq"],
        default="deepgram",
        help="Choose API: deepgram or groq",
    )
    parser.add_argument(
        "--topics", type=int, default=5, help="Number of topics for LDA model"
    )
    parser.add_argument("--groq-prompt", help="Optional prompt for Groq transcription")
    parser.add_argument(
        "--register",
        choices=["it-workflow", "gen-ai", "tech-support"],
        default="it-workflow",
        help="Analysis register: it-workflow, gen-ai, or tech-support",
    )
    parser.add_argument(
        "--skip-unsilence", action="store_true", help="Skip silence removal processing"
    )
    parser.add_argument(
        "--transcribe-only",
        action="store_true",
        help="Only perform audio transcription without topic modeling or video analysis",
    )
    parser.add_argument(
        "--software-list",
        type=str,
        help="Path to a text file containing list of software applications to detect (one per line)",
    )
    parser.add_argument(
        "--logo-db",
        type=str,
        help="Path to directory containing software logo templates",
    )
    parser.add_argument(
        "--ocr-lang", default="eng", help="Language for OCR detection (default: eng)"
    )
    parser.add_argument(
        "--logo-threshold",
        type=float,
        default=0.8,
        help="Confidence threshold for logo detection (0.0-1.0, default: 0.8)",
    )
    parser.add_argument(
        "--thumbnail-interval",
        type=int,
        default=5,
        help="Time interval between thumbnails in seconds (default: 5)",
    )
    parser.add_argument(
        "--max-thumbnails",
        type=int,
        default=5,
        help="Maximum number of thumbnails to generate per segment (default: 5)",
    )
    parser.add_argument(
        "--min-thumbnail-confidence",
        type=float,
        default=0.7,
        help="Minimum confidence threshold for thumbnail analysis (0.0-1.0, default: 0.7)",
    )

    parser.add_argument(
        "--analyze-screenshot",
        action="store_true",
        help="Analyze a single screenshot instead of processing video",
    )
    parser.add_argument(
        "--screenshot-context",
        help="Optional context to consider during screenshot analysis",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Validate input (file or YouTube URL)
    error, is_youtube = validate_input(args.input, args.analyze_screenshot)
    if error:
        print(f"Error: {error}")
        sys.exit(1)

    # Create project folder
    try:
        project_path = create_project_folder(args.input, args.output)
        print(f"Project folder: {project_path}")
    except Exception as e:
        print(f"Error creating project folder: {str(e)}")
        sys.exit(1)

    try:
        checkpoint = load_checkpoint(project_path)
        if checkpoint and checkpoint["stage"] == CHECKPOINTS["PROCESS_COMPLETE"]:
            results = checkpoint["data"]["results"]
            print("Loading results from previous complete run.")
        elif args.analyze_screenshot:
            # Read software list if provided
            software_list = None
            if args.software_list:
                if not os.path.exists(args.software_list):
                    print(f"Error: Software list file not found: {args.software_list}")
                    sys.exit(1)
                try:
                    with open(args.software_list, "r") as f:
                        software_list = [line.strip() for line in f if line.strip()]
                    print(
                        f"Loaded {len(software_list)} software applications to detect"
                    )
                except Exception as e:
                    print(f"Error reading software list file: {str(e)}")
                    sys.exit(1)

            from .analysis.visual_analysis import analyze_screenshot

            results = analyze_screenshot(
                args.input,
                project_path,
                software_list=software_list,
                logo_db_path=args.logo_db,
                ocr_lang=args.ocr_lang,
                logo_threshold=args.logo_threshold,
                context=args.screenshot_context,
            )

            print("\nScreenshot analysis complete.")
            print(f"Results saved in: {os.path.join(project_path, 'results.json')}")

            if software_list:
                print("\nSoftware Detection Results:")
                if results.get("software_detections"):
                    for detection in results["software_detections"]:
                        if detection.get("ocr_matches"):
                            print(
                                "\nText detected: "
                                + ", ".join(
                                    f"{m['software']} ({m['detected_text']})"
                                    for m in detection["ocr_matches"]
                                )
                            )
                        if detection.get("logo_matches"):
                            print(
                                "\nLogos detected: "
                                + ", ".join(
                                    f"{m['software']} (confidence: {m['confidence']:.2f})"
                                    for m in detection["logo_matches"]
                                )
                            )
                else:
                    print("No software detected in the screenshot.")

            if results.get("gemini_analysis"):
                print("\nGemini Analysis:")
                print(results["gemini_analysis"])

        elif args.input.endswith(".json"):
            transcript = load_transcript(
                args.input
            )  # This will be removed - loading transcript should be handled differently if needed.
            results = process_transcript(
                transcript, project_path, args.topics
            )  # Corrected import path
            print(
                "Note: Video splitting and Gemini analysis are not performed when processing a transcript file."
            )
        else:
            # Read software list if provided
            software_list = None
            if args.software_list:
                if not os.path.exists(args.software_list):
                    print(f"Error: Software list file not found: {args.software_list}")
                    sys.exit(1)
                try:
                    with open(args.software_list, "r") as f:
                        software_list = [line.strip() for line in f if line.strip()]
                    print(
                        f"Loaded {len(software_list)} software applications to detect"
                    )
                except Exception as e:
                    print(f"Error reading software list file: {str(e)}")
                    sys.exit(1)

            results = process_video(
                args.input,
                project_path,
                args.api,
                args.topics,
                args.groq_prompt,
                args.skip_unsilence,
                args.transcribe_only,
                is_youtube_url=is_youtube,
                software_list=software_list,
                logo_db_path=args.logo_db,
                ocr_lang=args.ocr_lang,
                logo_threshold=args.logo_threshold,
                thumbnail_interval=args.thumbnail_interval,
                max_thumbnails=args.max_thumbnails,
                min_thumbnail_confidence=args.min_thumbnail_confidence,
            )

        print(f"\nProcessing complete. Project folder: {project_path}")
        print(f"Results saved in: {os.path.join(project_path, 'results.json')}")

        if args.transcribe_only:
            print("\nTranscription completed successfully.")
            print("Transcript and raw transcription data saved in project folder.")
        else:
            if not args.analyze_screenshot:
                print("\nTop words for each topic:")
                for topic in results["topics"]:
                    print(f"Topic {topic['topic_id'] + 1}: {', '.join(topic['words'])}")

                print(
                    f"\nGenerated and analyzed {len(results['analyzed_segments'])} segments"
                )

                if args.software_list:
                    print("\nSoftware Detection Results:")
                    for segment in results["analyzed_segments"]:
                        if (
                            "software_detected" in segment
                            and segment["software_detected"]
                        ):
                            print(
                                f"\nSegment {segment['segment_id']} ({segment['start_time']:.2f}s - {segment['end_time']:.2f}s):"
                            )
                            print(f"Analysis: {segment['gemini_analysis']}")

                if not args.input.endswith(".json"):
                    print(
                        f"\nVideo segments saved in: {os.path.join(project_path, 'segments')}"
                    )

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Progress has been saved.")
        print(f"To resume, run the same command again with the same arguments.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred during processing: {str(e)}")
        print("Please check the project folder for any partial results or logs.")
        print(
            "You can resume processing by running the script again with the same arguments."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
