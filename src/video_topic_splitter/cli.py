"""Command-line interface for video topic splitter."""

import os
import sys
import argparse
from typing import Optional, Tuple
from dotenv import load_dotenv

from .core import process_video
from .youtube import is_youtube_url
from .project import create_project_folder, load_checkpoint
from .constants import CHECKPOINTS
from .transcription import load_transcript
from .topic_modeling import process_transcript

def validate_input(input_path: str) -> Tuple[Optional[str], bool]:
    """Validate input is either a valid file path or YouTube URL.
    
    Returns:
        Tuple[Optional[str], bool]: (error message if any, is_youtube_url)
    """
    # Check if input is a YouTube URL
    if is_youtube_url(input_path):
        return None, True
        
    # Check if input is a valid file
    if not os.path.exists(input_path):
        return f"Input file not found: {input_path}", False
    
    if not input_path.lower().endswith(('.mp4', '.mkv', '.json')):
        return f"Unsupported file format. Supported formats: .mp4, .mkv, .json", False
    
    return None, False

def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Process video or transcript for topic-based segmentation and multi-modal analysis"
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to input video file, YouTube URL, or transcript JSON",
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
        "--topics",
        type=int,
        default=5,
        help="Number of topics for LDA model"
    )
    parser.add_argument(
        "--groq-prompt",
        help="Optional prompt for Groq transcription"
    )
    parser.add_argument(
        "--skip-unsilence",
        action="store_true",
        help="Skip silence removal processing"
    )
    parser.add_argument(
        "--transcribe-only",
        action="store_true",
        help="Only perform audio transcription without topic modeling or video analysis"
    )
    
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Validate input (file or YouTube URL)
    error, is_youtube = validate_input(args.input)
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
        if checkpoint and checkpoint['stage'] == CHECKPOINTS['PROCESS_COMPLETE']:
            results = checkpoint['data']['results']
            print("Loading results from previous complete run.")
        elif args.input.endswith(".json"):
            transcript = load_transcript(args.input)
            results = process_transcript(transcript, project_path, args.topics)
            print(
                "Note: Video splitting and Gemini analysis are not performed when processing a transcript file."
            )
        else:
            results = process_video(
                args.input,
                project_path,
                args.api,
                args.topics,
                args.groq_prompt,
                args.skip_unsilence,
                args.transcribe_only,
                is_youtube_url=is_youtube
            )

        print(f"\nProcessing complete. Project folder: {project_path}")
        print(f"Results saved in: {os.path.join(project_path, 'results.json')}")
        
        if args.transcribe_only:
            print("\nTranscription completed successfully.")
            print("Transcript and raw transcription data saved in project folder.")
        else:
            print("\nTop words for each topic:")
            for topic in results["topics"]:
                print(f"Topic {topic['topic_id'] + 1}: {', '.join(topic['words'])}")
            print(f"\nGenerated and analyzed {len(results['analyzed_segments'])} segments")

            if not args.input.endswith(".json"):
                print(f"Video segments saved in: {os.path.join(project_path, 'segments')}")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Progress has been saved.")
        print(f"To resume, run the same command again with the same arguments.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred during processing: {str(e)}")
        print("Please check the project folder for any partial results or logs.")
        print("You can resume processing by running the script again with the same arguments.")
        sys.exit(1)

if __name__ == "__main__":
    main()
