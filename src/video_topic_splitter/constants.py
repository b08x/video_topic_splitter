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
    # Visual topic pipeline checkpoints
    "VISUAL_TOPIC_MODELING_COMPLETE": 8,
    "VISUAL_TOPIC_PIPELINE_COMPLETE": 9,
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

# Default software list for OCR detection
DEFAULT_SOFTWARE_LIST = [
    "VSCode", "Visual Studio Code", "Visual Studio", "PyCharm", "IntelliJ", 
    "Eclipse", "Sublime Text", "Atom", "Vim", "Emacs", "Notepad++",
    "Terminal", "Command Prompt", "PowerShell", "Bash", "Git Bash",
    "Chrome", "Firefox", "Safari", "Edge", "Opera",
    "Photoshop", "Illustrator", "GIMP", "Inkscape", "Figma", "Sketch",
    "Excel", "Word", "PowerPoint", "Google Sheets", "Google Docs", "Google Slides",
    "Jupyter", "Jupyter Notebook", "Colab", "Google Colab",
    "Docker", "Kubernetes", "AWS", "Azure", "GCP", "Google Cloud",
    "Unity", "Unreal Engine", "Blender", "Maya", "3DS Max",
    "Audacity", "Adobe Audition", "Logic Pro", "Pro Tools",
    "Final Cut Pro", "Adobe Premiere", "DaVinci Resolve"
]

# Default OCR language
DEFAULT_OCR_LANG = "eng"

# Default frame format
DEFAULT_FRAME_FORMAT = "jpg"

# Default compression quality
DEFAULT_COMPRESSION_QUALITY = 90

# Default frames per scene
DEFAULT_FRAMES_PER_SCENE = 3

# Note: LOGO_DB_PATH is removed as logo detection is being removed.

