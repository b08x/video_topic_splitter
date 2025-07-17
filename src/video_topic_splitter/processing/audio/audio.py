#!/usr/bin/env python3
"""Audio processing functionality for video topic splitter."""

import logging
import os
import subprocess

import ffmpeg

logger = logging.getLogger(__name__)


def extract_audio(video_path: str, output_path: str):
    """Extract audio from video file using ffmpeg-python."""
    try:
        (
            ffmpeg.input(video_path)
            .output(output_path, acodec="libopus", audio_bitrate="192k")
            .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
        )
    except ffmpeg.Error as e:
        logger.error("Error during audio extraction:")
        logger.error(e.stderr.decode())
        raise


def convert_to_mono_and_resample(
    input_path: str, output_path: str, sample_rate: int = 16000
):
    """Convert audio to mono and resample using ffmpeg-python."""
    try:
        (
            ffmpeg.input(input_path)
            .output(
                output_path,
                ac=1,  # Mono
                ar=str(sample_rate),
                acodec="aac",
                audio_bitrate="128k",
                # Apply a high-pass filter to remove low-frequency noise
                af="highpass=f=200",
            )
            .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
        )
        return {"status": "success", "message": "Audio converted successfully."}
    except ffmpeg.Error as e:
        logger.error("Error during audio conversion:")
        logger.error(e.stderr.decode())
        return {"status": "error", "message": e.stderr.decode()}


def normalize_audio(input_path: str, output_path: str) -> dict:
    """Normalize audio volume using ffmpeg-normalize."""
    try:
        command = [
            "ffmpeg-normalize",
            input_path,
            "-o",
            output_path,
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-f",
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        return {"status": "success", "message": "Audio normalized successfully."}
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg-normalize failed: {e.stderr}")
        return {"status": "error", "message": e.stderr}


def remove_silence(input_path: str, output_path: str) -> dict:
    """Remove silent parts of a video/audio file using the unsilence tool."""
    try:
        command = ["unsilence", input_path, output_path, "-af", "1.5s", "-a", "0.1"]
        subprocess.run(command, check=True, capture_output=True, text=True)
        return {"status": "success", "message": "Silence removal complete."}
    except subprocess.CalledProcessError as e:
        logger.error(f"Unsilence failed: {e.stderr}")
        return {"status": "error", "message": e.stderr}
    except FileNotFoundError:
        msg = "Unsilence command not found. Please ensure it is installed and in your PATH."
        logger.error(msg)
        return {"status": "error", "message": msg}