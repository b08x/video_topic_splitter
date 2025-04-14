#!/usr/bin/env python3
"""
Constants used throughout the refactored video scene splitter application.
"""

# Define checkpoint stages for the refactored pipeline
CHECKPOINTS = {
    "PROJECT_CREATED": 0,
    "YOUTUBE_DOWNLOAD_COMPLETE": 1,
    "AUDIO_PROCESSED": 2,
    "TRANSCRIPTION_COMPLETE": 3,
    "SCENES_DETECTED": 4,         # Changed from TOPIC_MODELING_COMPLETE
    "SCENE_ANALYSIS_COMPLETE": 5, # Changed from VIDEO_ANALYZED / VISUAL_ANALYSIS_COMPLETE
    "VIDEO_SPLIT_COMPLETE": 6,    # New stage for actual splitting output
    "PROCESS_COMPLETE": 7,
    # Removed: TOPIC_MODELING_COMPLETE, SEGMENTS_IDENTIFIED, VIDEO_ANALYZED, VISUAL_ANALYSIS_COMPLETE
    # Kept screenshot analysis separate if that mode is desired later, but focusing on video pipeline now.
    "SCREENSHOT_ANALYZED": 10,
}
"""
Dictionary mapping checkpoint names (stages) to integer values for the
scene-based video splitting pipeline.
"""

# Define a separate constant for the no scenes detected checkpoint
NO_SCENES_DETECTED = 4.5
"""
Checkpoint value for when no scenes are detected in the video.
This is between SCENES_DETECTED and SCENE_ANALYSIS_COMPLETE.
"""

# Note: LOGO_DB_PATH is removed as logo detection is being removed.

