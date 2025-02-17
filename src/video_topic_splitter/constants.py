#!/usr/bin/env python3
"""Constants used throughout the video topic splitter."""

# Define checkpoint stages
CHECKPOINTS = {
    "PROJECT_CREATED": 0,
    "YOUTUBE_DOWNLOAD_COMPLETE": 1,  # New checkpoint for YouTube downloads
    "AUDIO_PROCESSED": 2,
    "TRANSCRIPTION_COMPLETE": 3,
    "TRANSCRIBE_ONLY_COMPLETE": 4,  # Checkpoint for transcription-only mode
    "TOPIC_MODELING_COMPLETE": 5,
    "SEGMENTS_IDENTIFIED": 6,
    "VIDEO_ANALYZED": 7,
    "SCREENSHOT_ANALYZED": 8,  # New checkpoint for screenshot analysis
    "PROCESS_COMPLETE": 9,
}

LOGO_DB_PATH = "video_topic_splitter/data/logos"  # Path relative to the project root in final package
