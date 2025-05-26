# cli.py
#!/usr/bin/env python3
"""
Command-line interface for the video segmentation and analysis tool.
"""

import argparse
import logging
import os
from dotenv import load_dotenv


# Corrected import using relative path
from .core import VideoProcessor
from .api.gemini import GeminiClient # Assuming api is also a sibling directory/module

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """
    Main function to parse command-line arguments and run the video processing.
    """
    parser = argparse.ArgumentParser(
        description="Segment and analyze videos using the Gemini API."
    )
    parser.add_argument("video_path", help="Path to the input video file.")
    parser.add_argument("output_dir", help="Directory to store output files.")
    parser.add_argument(
        "--no-scene-detect",
        action="store_true",
        help="Disable scene detection as a pre-processing step.",
    )
    args = parser.parse_args()

    video_path = args.video_path
    output_dir = args.output_dir
    use_scene_detection = not args.no_scene_detect

    # Validate input paths
    if not os.path.exists(video_path):
        logging.error(f"Video file not found: {video_path}")
        return
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the Gemini client
    try:
        # Assuming GeminiClient is also part of the package
        gemini_client = GeminiClient(api_key=os.getenv("GEMINI_API_KEY"))
    except ValueError as e:
        logging.error(f"Failed to initialize Gemini client: {e}")
        return
    except RuntimeError as e:
        logging.error(f"Failed to initialize Gemini client: {e}")
        return

    # Initialize the VideoProcessor
    processor = VideoProcessor(gemini_client)

    # Run the video processing pipeline
    processor.process_video(video_path, output_dir, use_scene_detection)

    logging.info("Video processing complete.")


if __name__ == "__main__":
    main()
