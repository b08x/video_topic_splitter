# __init__.py
"""
Video Topic Splitter - A tool for segmenting videos based on topic analysis.
"""

from video_topic_splitter.core import process_video
from video_topic_splitter.project import (create_project_folder,
                                          load_checkpoint, save_checkpoint)

__version__ = "0.2.0"
