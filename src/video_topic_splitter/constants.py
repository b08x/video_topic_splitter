#!/usr/bin/env python3
"""Constants used throughout the video topic splitter."""

# Define checkpoint stages
CHECKPOINTS = {
    'PROJECT_CREATED': 0,
    'AUDIO_PROCESSED': 1,
    'TRANSCRIPTION_COMPLETE': 2,
    'TOPIC_MODELING_COMPLETE': 3,
    'SEGMENTS_IDENTIFIED': 4,
    'VIDEO_ANALYZED': 5,
    'PROCESS_COMPLETE': 6
}