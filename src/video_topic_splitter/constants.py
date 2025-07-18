#!/usr/bin/env python3
"""
Constants used throughout the video topic splitter application.

This module defines centralized constants for managing processing stages,
ensuring consistency across different modules.

Attributes:
    CHECKPOINTS (dict): A dictionary mapping human-readable checkpoint stage
        names to integer values. This allows the application to save and
        resume processing from a known state.
"""

# Define checkpoint stages
CHECKPOINTS = {
    "PROJECT_CREATED": 0,
    "YOUTUBE_DOWNLOAD_COMPLETE": 1,  # New checkpoint for YouTube downloads
    "AUDIO_PROCESSED": 2,
    "TRANSCRIPTION_COMPLETE": 3,
    "TOPIC_MODELING_COMPLETE": 5,
    "SEGMENTS_IDENTIFIED": 6,
    "SCENES_DETECTED": 7,  # New checkpoint for scene detection
    "VIDEO_ANALYZED": 8,
    "VISUAL_ANALYSIS_COMPLETE": 9,  # Checkpoint for visual analysis completion
    "SCREENSHOT_ANALYZED": 10,  # New checkpoint for screenshot analysis
    "PROCESS_COMPLETE": 11,
}


