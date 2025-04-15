#!/usr/bin/env python3
"""
Constants used throughout the refactored video scene splitter application.
"""

# Proposed new checkpoint stages for the unified pipeline
CHECKPOINTS = {
    "PROJECT_CREATED": 0,
    "YOUTUBE_DOWNLOAD_COMPLETE": 1,
    "AUDIO_PROCESSED": 2,
    "TRANSCRIPTION_COMPLETE": 3,
    "SCENES_DETECTED": 4,
    "VISUAL_ANALYSIS_COMPLETE": 5,     # Frames analyzed by Gemini/OCR
    "VISUAL_TOPIC_MODELING_COMPLETE": 6, # Text+Visual analysis via VisualTopicAnalyzer
    "VIDEO_SPLIT_COMPLETE": 7,         # Video split based on final segments
    "PROCESS_COMPLETE": 8,
    # Consider removing or re-evaluating these:
    # "SCREENSHOT_ANALYZED": 10, # Keep if screenshot mode is still desired separately
}
"""
Dictionary mapping checkpoint names (stages) to integer values for the
unified video processing pipeline.
"""

# Keep NO_SCENES_DETECTED if needed, adjust value if necessary
NO_SCENES_DETECTED = 4.5 # Falls between SCENES_DETECTED and VISUAL_ANALYSIS_COMPLETE
"""
Checkpoint value for when no scenes are detected in the video.
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