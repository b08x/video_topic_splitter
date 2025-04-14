# utils/youtube.py
#!/usr/bin/env python3
"""Provides utilities for interacting with YouTube.

This module includes functions for validating YouTube URLs, fetching video
metadata without downloading the full video, finding the best available
thumbnail URL, and downloading videos using yt-dlp. It integrates with
the ThumbnailManager for saving thumbnails during the download process.
"""

import logging
import os
import re
from typing import Dict, List, Optional

import yt_dlp

from .thumbnail import ThumbnailManager

logger = logging.getLogger(__name__)

# YouTube URL validation regex
YOUTUBE_URL_PATTERN = (
    r"^(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[a-zA-Z0-9_-]{11}$"
)

# YouTube thumbnail quality options, ordered from highest to lowest
THUMBNAIL_QUALITIES = [
    "maxresdefault",  # 1920x1080
    "sddefault",  # 640x480
    "hqdefault",  # 480x360
    "mqdefault",  # 320x180
    "default",  # 120x90
]


def is_youtube_url(url: str) -> bool:
    """Check if the given URL is a valid YouTube video URL.

    Uses a regular expression to match common YouTube video URL formats.

    Args:
        url: The URL string to validate.

    Returns:
        True if the URL matches the YouTube video pattern, False otherwise.
    """
    return bool(re.match(YOUTUBE_URL_PATTERN, url))


def get_video_info(url: str) -> Optional[Dict]:
    """Get video information (metadata) without downloading the video content.

    Uses yt-dlp's 'extract_info' with download=False to fetch metadata like
    title, duration, uploader, and available formats.

    Args:
        url: The YouTube video URL.

    Returns:
        A dictionary containing video metadata if successful, None if the URL
        is invalid or an error occurs during extraction.
    """
    if not is_youtube_url(url):
        logger.warning(f"Invalid YouTube URL provided: {url}")
        return None

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,  # Faster extraction, gets basic info
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            # yt-dlp might return a playlist structure even for single videos
            # when using extract_flat. Ensure we get the video entry.
            if info and "entries" in info:
                return info["entries"][0]
            return info
    except yt_dlp.utils.DownloadError as e:
        logger.error(f"yt-dlp error getting video info for {url}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error getting video info for {url}: {str(e)}")
        return None


def get_best_thumbnail_url(video_info: Dict) -> Optional[str]:
    """Get the best quality thumbnail URL available for a given video ID.

    Iterates through a predefined list of thumbnail quality options
    (from highest to lowest resolution) and checks if the thumbnail URL exists
    by sending a HEAD request.

    Args:
        video_info: A dictionary containing video metadata, expected to have
                    an 'id' key with the YouTube video ID.

    Returns:
        The URL string of the highest quality available thumbnail, or None if
        no valid thumbnail URL is found or if video_info is invalid.
    """
    if not video_info or "id" not in video_info:
        logger.warning("Invalid video_info provided to get_best_thumbnail_url.")
        return None

    video_id = video_info["id"]
    # Try standard thumbnail URLs first
    for quality in THUMBNAIL_QUALITIES:
        url = f"https://img.youtube.com/vi/{video_id}/{quality}.jpg"
        try:
            import requests  # Import locally to avoid making it a hard dependency

            # Use HEAD request for efficiency, we only need to check existence
            response = requests.head(url, timeout=5)
            if response.status_code == 200:
                logger.info(f"Found thumbnail for {video_id} at quality {quality}.")
                return url
        except requests.exceptions.RequestException as e:
            logger.debug(f"Error checking thumbnail {url}: {str(e)}")
            continue # Try next quality
        except Exception as e:
            logger.warning(f"Unexpected error checking thumbnail {url}: {str(e)}")
            continue # Try next quality

    # Fallback: Check if thumbnail info is directly in video_info (less common)
    if "thumbnail" in video_info:
        logger.info(f"Using thumbnail URL from video_info for {video_id}.")
        return video_info["thumbnail"]

    logger.warning(f"Could not find any valid thumbnail for video ID {video_id}.")
    return None


def download_video(url: str, output_path: str, project_path: str = None) -> dict:
    """Download a YouTube video using yt-dlp and optionally save its thumbnail.

    Downloads the best available MP4 format (video + audio). If a project_path
    is provided, it attempts to find and save the best quality thumbnail
    using the ThumbnailManager.

    Args:
        url: The YouTube video URL.
        output_path: The full path (including filename) where the downloaded
                     video should be saved.
        project_path: Optional path to the project directory. If provided,
                      thumbnails will be saved within a 'thumbnails' subfolder
                      in this directory.

    Returns:
        A dictionary containing the download status:
        - {"status": "success", "message": str, "file_path": str,
           "video_info": dict, "thumbnail_info": Optional[dict]}
        - {"status": "error", "message": str}
    """
    if not is_youtube_url(url):
        return {"status": "error", "message": "Invalid YouTube URL"}

    try:
        # 1. Get video info and thumbnail URL
        video_info = get_video_info(url)
        if not video_info:
            return {"status": "error", "message": "Failed to get video information"}

        thumbnail_info = None
        if project_path:
            thumbnail_url = get_best_thumbnail_url(video_info)
            if thumbnail_url:
                try:
                    thumbnail_manager = ThumbnailManager(project_path)
                    thumbnail_info = thumbnail_manager.save_youtube_thumbnail(
                        thumbnail_url
                    )
                    if thumbnail_info:
                        logger.info(
                            f"Saved YouTube thumbnail to: {thumbnail_info['path']}"
                        )
                    else:
                        logger.warning(
                            f"Failed to save YouTube thumbnail from {thumbnail_url}"
                        )
                except Exception as e:
                    logger.error(f"Error initializing or using ThumbnailManager: {e}")
            else:
                logger.warning("Could not find a suitable thumbnail URL to download.")

        # 2. Configure yt-dlp for download
        # Ensure directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir: # Handle cases where output_path is just a filename
            os.makedirs(output_dir, exist_ok=True)

        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": output_path,
            "merge_output_format": "mp4",
            "postprocessors": [
                {
                    "key": "FFmpegVideoConvertor",
                    "preferedformat": "mp4",
                }
            ],
            "quiet": False, # Show download progress
            "no_warnings": False,
            "progress_hooks": [lambda d: print(f"Download status: {d['status']}") if d['status'] in ['downloading', 'finished'] else None],
        }

        # 3. Perform download
        logger.info(f"Starting download for {url} to {output_path}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # 4. Verify download and return result
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info(f"Video downloaded successfully to {output_path}")
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
            # Check if yt-dlp created a file with a different extension (unlikely with opts)
            base, _ = os.path.splitext(output_path)
            possible_files = [f for f in os.listdir(output_dir or '.') if f.startswith(os.path.basename(base))]
            if possible_files:
                 logger.warning(f"Downloaded file might have unexpected name: {possible_files[0]}. Expected: {output_path}")
                 # Attempt to rename or just report error
                 # For simplicity, reporting error for now.
            return {
                "status": "error",
                "message": f"Download process finished but output file not found or empty: {output_path}",
            }

    except yt_dlp.utils.DownloadError as e:
        logger.error(f"yt-dlp download error for {url}: {str(e)}")
        return {"status": "error", "message": f"Error downloading video: {str(e)}"}
    except Exception as e:
        logger.exception(f"Unexpected error downloading video {url}: {str(e)}") # Use exception for stack trace
        return {"status": "error", "message": f"Unexpected error downloading video: {str(e)}"}
