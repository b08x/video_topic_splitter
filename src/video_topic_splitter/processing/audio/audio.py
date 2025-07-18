#!/usr/bin/env python3
"""Audio processing functionality for video topic splitter."""

import logging
import os
import subprocess

import ffmpeg
from ...progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)


def extract_audio(video_path: str, output_path: str, progress_tracker: ProgressTracker = None):
    """
    Extract the audio track from a video file and save it as an Opus audio file.

    This function uses ffmpeg to perform the extraction, which is efficient and
    handles a wide variety of video formats.

    Args:
        video_path: The path to the input video file.
        output_path: The path where the extracted audio file will be saved.
        progress_tracker: An optional ProgressTracker instance to report progress.

    Raises:
        ffmpeg.Error: If ffmpeg encounters an error during extraction.
    """
    if progress_tracker:
        progress_tracker.update_phase_progress(0.0, "Extracting audio from video...")
    
    try:
        (
            ffmpeg.input(video_path)
            .output(output_path, acodec="libopus", audio_bitrate="192k")
            .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
        )
        
        if progress_tracker:
            progress_tracker.update_phase_progress(100.0, "Audio extraction complete")
    except ffmpeg.Error as e:
        logger.error("Error during audio extraction:")
        logger.error(e.stderr.decode())
        if progress_tracker:
            progress_tracker.fail_phase(f"Audio extraction failed: {e.stderr.decode()}")
        raise


def convert_to_mono_and_resample(
    input_path: str, output_path: str, sample_rate: int = 16000, progress_tracker: ProgressTracker = None
):
    """
    Convert an audio file to mono and resample it to a specific sample rate.

    This is often a required preprocessing step for speech-to-text models.
    It also applies a high-pass filter to remove low-frequency noise.

    Args:
        input_path: The path to the input audio file.
        output_path: The path where the converted audio file will be saved.
        sample_rate: The target sample rate in Hz.
        progress_tracker: An optional ProgressTracker instance to report progress.

    Returns:
        A dictionary with the status of the conversion.
    """
    if progress_tracker:
        progress_tracker.update_phase_progress(0.0, "Converting audio to mono and resampling...")
    
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
        
        if progress_tracker:
            progress_tracker.update_phase_progress(100.0, "Audio conversion complete")
        return {"status": "success", "message": "Audio converted successfully."}
    except ffmpeg.Error as e:
        logger.error("Error during audio conversion:")
        logger.error(e.stderr.decode())
        if progress_tracker:
            progress_tracker.fail_phase(f"Audio conversion failed: {e.stderr.decode()}")
        return {"status": "error", "message": e.stderr.decode()}


def normalize_audio(input_path: str, output_path: str, progress_tracker: ProgressTracker = None) -> dict:
    """
    Normalize the volume of an audio file to a standard level.

    This uses the `ffmpeg-normalize` utility to ensure consistent audio levels,
    which can improve the quality of transcription.

    Args:
        input_path: The path to the input audio file.
        output_path: The path where the normalized audio file will be saved.
        progress_tracker: An optional ProgressTracker instance to report progress.

    Returns:
        A dictionary with the status of the normalization.
    """
    if progress_tracker:
        progress_tracker.update_phase_progress(0.0, "Normalizing audio volume...")
    
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
        
        if progress_tracker:
            progress_tracker.update_phase_progress(100.0, "Audio normalization complete")
        return {"status": "success", "message": "Audio normalized successfully."}
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg-normalize failed: {e.stderr}")
        if progress_tracker:
            progress_tracker.fail_phase(f"Audio normalization failed: {e.stderr}")
        return {"status": "error", "message": e.stderr}


def remove_silence(input_path: str, output_path: str, progress_tracker: ProgressTracker = None) -> dict:
    """
    Remove silent portions from an audio or video file.

    This uses the `unsilence` command-line tool to detect and remove periods
    of silence, which can help to create a more concise audio track for
    analysis.

    Args:
        input_path: The path to the input audio or video file.
        output_path: The path where the processed file will be saved.
        progress_tracker: An optional ProgressTracker instance to report progress.

    Returns:
        A dictionary with the status of the silence removal process.
    """
    if progress_tracker:
        progress_tracker.update_phase_progress(0.0, "Removing silence from audio...")
    
    try:
        command = ["unsilence", input_path, output_path, "-af", "1.5s", "-a", "0.1"]
        subprocess.run(command, check=True, capture_output=True, text=True)
        
        if progress_tracker:
            progress_tracker.update_phase_progress(100.0, "Silence removal complete")
        return {"status": "success", "message": "Silence removal complete."}
    except subprocess.CalledProcessError as e:
        logger.error(f"Unsilence failed: {e.stderr}")
        if progress_tracker:
            progress_tracker.fail_phase(f"Silence removal failed: {e.stderr}")
        return {"status": "error", "message": e.stderr}
    except FileNotFoundError:
        msg = "Unsilence command not found. Please ensure it is installed and in your PATH."
        logger.error(msg)
        if progress_tracker:
            progress_tracker.fail_phase(msg)
        return {"status": "error", "message": msg}