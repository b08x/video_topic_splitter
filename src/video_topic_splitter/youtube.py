#!/usr/bin/env python3
"""YouTube video downloading and processing functionality."""

import logging
import os
import re
from typing import Dict, List, Optional

import yt_dlp

from .thumbnail_utils import ThumbnailManager

logger = logging.getLogger(__name__)

# YouTube URL validation regex
YOUTUBE_URL_PATTERN = (
    r"^(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[a-zA-Z0-9_-]{11}$"
)

# YouTube thumbnail quality options
THUMBNAIL_QUALITIES = [
    "maxresdefault",  # 1920x1080
    "sddefault",  # 640x480
    "hqdefault",  # 480x360
    "mqdefault",  # 320x180
    "default",  # 120x90
]


def is_youtube_url(url: str) -> bool:
    """Check if the given URL is a valid YouTube video URL.

    Args:
        url (str): The URL to check.

    Returns:
        bool: True if the URL is a valid YouTube video URL, False otherwise.
    """
    return bool(re.match(YOUTUBE_URL_PATTERN, url))


def get_video_info(url: str) -> Optional[Dict]:
    """Get video information from a YouTube URL without downloading the video.

    Args:
        url (str): The YouTube video URL.

    Returns:
        Optional[Dict]: A dictionary containing video information if successful, None otherwise.  Includes error handling for invalid URLs and API failures.
    """
    if not is_youtube_url(url):
        return None

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info
    except Exception as e:
        logger.error(f"Error getting video info: {str(e)}")
        return None


def get_best_thumbnail_url(video_info: Dict) -> Optional[str]:
    """Get the highest quality thumbnail URL available for a given video.

    Args:
        video_info (Dict): A dictionary containing video information (as returned by get_video_info).

    Returns:
        Optional[str]: The URL of the best quality thumbnail, or None if no thumbnail is found.  Handles potential network errors.
    """
    if not video_info:
        return None

    # Try each quality option in descending order of preference
    for quality in THUMBNAIL_QUALITIES:
        url = f"https://img.youtube.com/vi/{video_info['id']}/{quality}.jpg"
        try:
            import requests

            response = requests.head(url)
            if response.status_code == 200:
                return url
        except Exception:
            continue

    return None


def download_video(url: str, output_path: str, project_path: str = None) -> Dict:
    """Download a YouTube video using yt-dlp and optionally save a thumbnail.

    Args:
        url (str): The YouTube video URL.
        output_path (str): The path to save the downloaded video file.
        project_path (str, optional): Optional project path for saving thumbnails. Defaults to None.

    Returns:
        Dict: A dictionary containing the status and message of the download operation.  Includes video information and thumbnail information if successful.  Provides detailed error messages.
    """
    if not is_youtube_url(url):
        return {"status": "error", "message": "Invalid YouTube URL"}

    try:
        # Get video info and thumbnail URL
        video_info = get_video_info(url)
        if not video_info:
            return {"status": "error", "message": "Failed to get video information"}

        # Save thumbnail if project path is given
        thumbnail_info = None
        if project_path:
            thumbnail_url = get_best_thumbnail_url(video_info)
            if thumbnail_url:
                thumbnail_manager = ThumbnailManager(project_path)
                thumbnail_info = thumbnail_manager.save_youtube_thumbnail(thumbnail_url)

        # Download video using yt-dlp with specified format options
        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
            "outtmpl": output_path,
            "merge_output_format": "mp4",
            "postprocessors": [
                {
                    "key": "FFmpegVideoConvertor",
                    "preferedformat": "mp4",
                }
            ],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Check if download was successful and return results
        if os.path.exists(output_path):
            result = {
                "status": "success",
                "message": f"Video downloaded successfully to {output_path}",
                "file_path": output_path,
                "video_info": video_info,
            }
            if thumbnail_info:
                result["thumbnail_info"] = thumbnail_info
            return result
        else:
            return {
                "status": "error",
                "message": "Download completed but file not found",
            }

    except Exception as e:
        return {"status": "error", "message": f"Error downloading video: {str(e)}"}
