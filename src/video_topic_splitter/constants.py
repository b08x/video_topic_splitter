#!/usr/bin/env python3
"""Constants used throughout the video topic splitter."""

# Define checkpoint stages
CHECKPOINTS = {
    'PROJECT_CREATED': 0,
    'AUDIO_PROCESSED': 1,
    'TRANSCRIPTION_COMPLETE': 2,
    'TRANSCRIBE_ONLY_COMPLETE': 3,  # New checkpoint for transcription-only mode
    'TOPIC_MODELING_COMPLETE': 4,
    'SEGMENTS_IDENTIFIED': 5,
    'VIDEO_ANALYZED': 6,
    'PROCESS_COMPLETE': 7
}
