# processing/video/video_analysis.py
#!/usr/bin/env python3
"""Video analysis and segmentation functionality."""


import json
import logging
import os

import cv2
import numpy as np
import progressbar
from moviepy.editor import VideoFileClip
from PIL import Image, UnidentifiedImageError

from ...analysis.visual_analysis import LOGO_DB_PATH, detect_software_logos
from ...api.gemini import analyze_with_gemini  # Corrected import path
from ...constants import CHECKPOINTS
from ...project import save_checkpoint
from ..ocr.ocr_detection import detect_software_names

logger = logging.getLogger(__name__)


# analyze_screenshot moved to analysis/visual_analysis.py
# analyze_frame_for_software moved to analysis/visual_analysis.py
# analyze_thumbnails moved to analysis/visual_analysis.py
# analyze_segment_with_gemini moved to analysis/visual_analysis.py
