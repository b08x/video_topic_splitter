#!/usr/bin/env python3
"""YouTube video downloading and processing functionality."""

import os
import re
import yt_dlp

# YouTube URL validation regex
YOUTUBE_URL_PATTERN = r'^(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[a-zA-Z0-9_-]{11}$'

def is_youtube_url(url: str) -> bool:
    """Check if the given URL is a valid YouTube video URL."""
    return bool(re.match(YOUTUBE_URL_PATTERN, url))

def download_video(url: str, output_path: str) -> dict:
    """Download YouTube video using yt-dlp.
    
    Args:
        url: YouTube video URL
        output_path: Path where the video should be saved
        
    Returns:
        dict: Status and message about the download
    """
    if not is_youtube_url(url):
        return {
            "status": "error",
            "message": "Invalid YouTube URL"
        }
        
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
        'outtmpl': output_path,
        'merge_output_format': 'mp4',
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }],
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            
        if os.path.exists(output_path):
            return {
                "status": "success",
                "message": f"Video downloaded successfully to {output_path}",
                "file_path": output_path
            }
        else:
            return {
                "status": "error",
                "message": "Download completed but file not found"
            }
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error downloading video: {str(e)}"
        }
