#!/usr/bin/env python3
"""
Constants used throughout the video topic splitter application.

This module defines various constants that are used across different parts of
the video topic splitter project. Centralizing these constants helps in
maintaining consistency and makes it easier to update values if needed.
"""

# Define checkpoint stages
CHECKPOINTS = {
    "PROJECT_CREATED": 0,
    "YOUTUBE_DOWNLOAD_COMPLETE": 1,  # New checkpoint for YouTube downloads
    "AUDIO_PROCESSED": 2,
    "TRANSCRIPTION_COMPLETE": 3,
    "TRANSCRIBE_ONLY_COMPLETE": 4,  # Checkpoint for transcription-only mode
    "TOPIC_MODELING_COMPLETE": 5,
    "SEGMENTS_IDENTIFIED": 6,
    "SCENES_DETECTED": 7,  # New checkpoint for scene detection
    "VIDEO_ANALYZED": 8,
    "VISUAL_ANALYSIS_COMPLETE": 9,  # Checkpoint for visual analysis completion
    "SCREENSHOT_ANALYZED": 10,  # New checkpoint for screenshot analysis
    "PROCESS_COMPLETE": 11,
}
"""
Dictionary mapping checkpoint names (stages) to integer values.

These checkpoints represent distinct stages in the video processing pipeline.
They are used to track the progress of a video processing job, allowing the
process to be potentially resumed from the last completed stage. The integer
values indicate the order of the stages.
"""

LOGO_DB_PATH = "video_topic_splitter/data/logos"  # Path relative to the project root in final package
"""
Relative path to the directory containing the logo database.

This path specifies the location of the logo image files used for logo detection
within the video frames. It is defined relative to the project's root directory
when the package is installed or run.
"""
