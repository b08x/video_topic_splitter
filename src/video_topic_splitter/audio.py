#!/usr/bin/env python3
"""Audio processing utilities for video topic splitter."""

"""Audio processing utilities."""

import os
import subprocess

from pydub import AudioSegment


def convert_to_mono_and_resample(input_file, output_file, sample_rate=16000):
    """Converts audio to mono, resamples, applies gain control, and a high-pass filter.

    Args:
        input_file (str): Path to the input audio file.
        output_file (str): Path to save the processed audio file.
        sample_rate (int, optional): Target sample rate. Defaults to 16000.

    Returns:
        dict: A dictionary containing the status and a message.
            The status can be "success" or "error".
    """
    try:
        command = [
            "ffmpeg",
            "-i",
            input_file,
            "-af",
            "highpass=f=200, acompressor=threshold=-20dB:ratio=2:attack=5:release=50",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            output_file,
        ]
        subprocess.run(command, check=True)
        return {
            "status": "success",
            "message": f"Audio converted to mono, resampled to {sample_rate}Hz, gain-adjusted, high-pass filtered, and saved to {output_file}",
        }
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Error during audio conversion: {str(e)}",
        }


def normalize_audio(input_file, output_file, lowpass_freq=6000, highpass_freq=100):
    """Normalizes audio using ffmpeg-normalize.

    Args:
        input_file (str): Path to the input audio file.
        output_file (str): Path to save the normalized audio file.
        lowpass_freq (int, optional): Lowpass filter frequency. Defaults to 6000.
        highpass_freq (int, optional): Highpass filter frequency. Defaults to 100.

    Returns:
        dict: A dictionary containing the status and a message.
            The status can be "success" or "error".
    """
    try:
        command = [
            "ffmpeg-normalize",
            "-pr",
            "-tp",
            "-9.0",
            "-nt",
            "rms",
            input_file,
            "-prf",
            f"highpass=f={highpass_freq}",
            "-prf",
            "dynaudnorm=p=0.4:s=15",
            "-pof",
            f"lowpass=f={lowpass_freq}",
            "-ar",
            "48000",
            "-c:a",
            "pcm_s16le",
            "--keep-loudness-range-target",
            "-o",
            output_file,
        ]
        subprocess.run(command, check=True)
        return {
            "status": "success",
            "message": f"Audio normalized and saved to {output_file}",
        }
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Error during audio normalization: {str(e)}",
        }


def remove_silence(input_file, output_file, duration="1.5", threshold="-20"):
    """Removes silence from audio using unsilence.

    Args:
        input_file (str): Path to the input audio file.
        output_file (str): Path to save the processed audio file.
        duration (str, optional): Minimum silence duration in seconds. Defaults to "1.5".
        threshold (str, optional): Silence threshold in dB. Defaults to "-20".

    Returns:
        dict: A dictionary containing the status and a message.
            The status can be "success" or "error".
    """
    try:
        command = [
            "unsilence",
            "-d",
            "-ss",
            duration,
            "-sl",
            threshold,
            input_file,
            output_file,
        ]
        subprocess.run(command, check=True)
        return {
            "status": "success",
            "message": f"Silence removed from audio and saved to {output_file}",
        }
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Error during silence removal: {str(e)}",
        }


def extract_audio(video_path, output_path):
    """Extract audio from video file.

    Args:
        video_path (str): Path to the video file.
        output_path (str): Path to save the extracted audio file.

    Raises:
        Exception: If audio extraction fails.
    """
    print("Extracting audio from video...")
    try:
        audio = AudioSegment.from_file(video_path)
        audio.export(output_path, format="wav")
        print("Audio extraction complete.")
    except Exception as e:
        print(f"Error during audio extraction: {str(e)}")
        raise
