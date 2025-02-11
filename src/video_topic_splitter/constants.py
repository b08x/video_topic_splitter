#!/usr/bin/env python3
"""Constants used throughout the video topic splitter."""

# Define checkpoint stages
CHECKPOINTS = {
    "PROJECT_CREATED": 0,
    "YOUTUBE_DOWNLOAD_COMPLETE": 1,  # New checkpoint for YouTube downloads
    "AUDIO_PROCESSED": 2,
    "TRANSCRIPTION_COMPLETE": 3,
    "PROSODIC_FEATURES_EXTRACTED": 4,  # Moved before topic modeling
    "FEATURES_ALIGNED": 5,  # Moved before topic modeling
    "TOPIC_MODELING_COMPLETE": 6,
    "SEGMENTS_IDENTIFIED": 7,
    "VIDEO_ANALYZED": 8,
    "SCREENSHOT_ANALYZED": 9,
    "SEMANTIC_EMBEDDINGS_GENERATED": 10,
    "COMBINED_ANALYSIS_COMPLETE": 11,
    "TRANSCRIBE_ONLY_COMPLETE": 12,  # Moved to end since it's an alternative path
    "PROCESS_COMPLETE": 13,
}
