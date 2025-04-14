# processing/audio/audio.py
#!/usr/bin/env python3

"""Audio processing utilities using external tools like ffmpeg and libraries like pydub."""

import logging
import os
import subprocess
from contextlib import contextmanager

import ffmpeg  # Note: This import seems unused directly, subprocess calls ffmpeg
from moviepy.editor import VideoFileClip
from pydub import AudioSegment  # Note: This import seems unused
from unsilence import Unsilence

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def convert_to_mono_and_resample(input_file, output_file, sample_rate=16000):
    """
    Converts an audio file to mono, resamples it, applies gain control,
    a high-pass filter, and compression using ffmpeg.

    This function processes the audio to make it suitable for tasks like
    speech recognition by standardizing the format and applying basic
    audio enhancements.

    Args:
        input_file (str): Path to the input audio file.
        output_file (str): Path where the processed audio file will be saved.
        sample_rate (int, optional): The target sample rate in Hz. Defaults to 16000.

    Returns:
        dict: A dictionary containing the status ('success' or 'error') and
              a message (stdout on success, stderr or error message on failure).
              Example: {'status': 'success', 'message': 'ffmpeg output...'}
                       {'status': 'error', 'message': 'ffmpeg error...'}

    Raises:
        FileNotFoundError: Logged critically if ffmpeg is not found in the system PATH.
        subprocess.CalledProcessError: Logged as error if ffmpeg command fails.
        Exception: Logged with traceback for any other unexpected errors.
    """
    try:
        command = [
            "ffmpeg",
            "-i",
            input_file,
            "-af",
            # Apply multiple filters: -3dB volume, highpass at 200Hz, compressor
            "volume=-3dB,highpass=f=200, acompressor=threshold=-20dB:ratio=2:attack=5:release=50",
            "-ar",
            str(sample_rate),  # Set audio sample rate
            "-ac",
            "1",  # Set audio channels to 1 (mono)
            "-c:a",
            "aac",  # Set audio codec to AAC
            "-b:a",
            "128k",  # Set audio bitrate
            output_file,
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logging.info(
            f"Audio converted to mono, resampled to {sample_rate}Hz, gain-adjusted, high-pass filtered, and saved to {output_file}"
        )
        return {"status": "success", "message": result.stdout}
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during audio conversion: {e.stderr}")
        return {"status": "error", "message": e.stderr}
    except FileNotFoundError:
        logging.critical(
            f"ffmpeg not found. Please ensure it is installed and in your PATH."
        )
        return {"status": "error", "message": "ffmpeg not found"}
    except Exception as e:
        logging.exception(f"An unexpected error occurred during audio conversion: {e}")
        return {"status": "error", "message": str(e)}


def normalize_audio(input_file, output_file, lowpass_freq=6000, highpass_freq=100):
    """
    Normalizes audio loudness using ffmpeg-normalize and applies filters.

    This function uses the ffmpeg-normalize tool to adjust the audio level
    to a standard RMS target, applies dynamic range compression, and
    high/low-pass filters. It re-encodes the audio, choosing the codec
    based on the output file extension.

    Args:
        input_file (str): Path to the input audio file.
        output_file (str): Path where the normalized audio file will be saved.
                           The extension determines the output codec (.mp4/.m4a -> opus, others -> pcm_s16le).
        lowpass_freq (int, optional): Frequency for the low-pass filter. Defaults to 6000.
        highpass_freq (int, optional): Frequency for the high-pass filter. Defaults to 100.

    Returns:
        dict: A dictionary containing the status ('success' or 'error') and
              a message (stdout on success, stderr or error message on failure).
              Example: {'status': 'success', 'message': 'ffmpeg-normalize output...'}
                       {'status': 'error', 'message': 'ffmpeg-normalize error...'}

    Raises:
        FileNotFoundError: Logged critically if ffmpeg-normalize is not found.
        subprocess.CalledProcessError: Logged as error if ffmpeg-normalize command fails.
        Exception: Logged with traceback for any other unexpected errors.
    """
    output_ext = os.path.splitext(output_file)[1].lower()

    try:
        command = [
            "ffmpeg-normalize",
            "-pr",  # Enable progress report
            "-tp",
            "-9.0",  # Target peak level (dBFS)
            "-nt",
            "rms",  # Normalization type (Root Mean Square)
            input_file,
            "-prf",
            f"volume=-3dB,highpass=f={highpass_freq}",  # Pre-normalization filter
            "-prf",
            "dynaudnorm=p=0.4:s=15",  # Dynamic audio normalizer pre-filter
            "-pof",
            f"lowpass=f={lowpass_freq}",  # Post-normalization filter
            "-ar",
            "48000",  # Set output sample rate
        ]

        # Add codec settings based on output format
        if output_ext in [".mp4", ".m4a"]:
            command.extend(
                [
                    "-c:a",
                    "aac",  # Use Opus codec for MP4/M4A
                ]
            )
        else:
            command.extend(
                [
                    "-c:a",
                    "pcm_s16le",  # Use PCM S16LE for other formats (like WAV)
                ]
            )

        command.extend(
            [
                "--keep-loudness-range-target", # Maintain loudness range target
                "-o",
                output_file, # Output file path
            ]
        )
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logging.info(f"Audio normalized and saved to {output_file}")
        return {"status": "success", "message": result.stdout}
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during audio normalization: {e.stderr}")
        return {"status": "error", "message": e.stderr}
    except FileNotFoundError:
        logging.critical(
            f"ffmpeg-normalize not found. Please ensure it is installed and in your PATH."
        )
        return {"status": "error", "message": "ffmpeg-normalize not found"}
    except Exception as e:
        logging.exception(
            f"An unexpected error occurred during audio normalization: {e}"
        )
        return {"status": "error", "message": str(e)}


@contextmanager
def changed_working_directory(new_dir):
    """
    A context manager to temporarily change the current working directory.

    This is useful when a library or external tool requires running from a
    specific directory. Upon exiting the `with` block, the original working
    directory is restored, even if errors occur within the block.

    Args:
        new_dir (str): The path to the directory to change into temporarily.

    Yields:
        None: The context manager yields control to the `with` block.

    Example:
        >>> with changed_working_directory('/path/to/other/dir'):
        ...     # Code here runs with '/path/to/other/dir' as cwd
        ...     print(os.getcwd())
        >>> # Outside the block, the original cwd is restored
        >>> print(os.getcwd())
    """
    old_dir = os.getcwd()
    try:
        os.chdir(new_dir)
        yield
    finally:
        os.chdir(old_dir)


def remove_silence(input_file, output_file, audible_speed=2, silent_speed=8):
    """
    Adjusts audio speed, speeding up silent parts more than audible parts,
    using the 'unsilence' library.

    This function detects silence in the input audio and renders a new audio
    file where silent segments are sped up significantly more than audible
    segments, effectively shortening the total duration while preserving
    the audible content at a faster pace. It requires the 'unsilence'
    library and its dependencies (like ffmpeg).

    Note: 'unsilence' works by operating in the directory of the input file.
    This function temporarily changes the working directory for compatibility.

    Args:
        input_file (str): Path to the input audio file.
        output_file (str): Path where the silence-adjusted audio file will be saved.
                           The directory will be created if it doesn't exist.
        audible_speed (float, optional): Speed multiplier for audible segments.
                                         Defaults to 2 (2x speed).
        silent_speed (float, optional): Speed multiplier for silent segments.
                                        Defaults to 8 (8x speed).

    Returns:
        dict: A dictionary containing the status ('success' or 'error') and
              a message (empty on success, error message on failure).
              Example: {'status': 'success', 'message': ''}
                       {'status': 'error', 'message': 'Error details...'}

    Raises:
        Exception: Logs any exception that occurs during the process.
                   The specific exceptions depend on the 'unsilence' library
                   and file system operations.
    """
    try:
        # Get the directory for the output file (creating parent dirs if needed)
        output_dir = os.path.dirname(output_file)
        os.makedirs(
            output_dir, exist_ok=True
        )  # Create output directory if it doesn't exist

        input_dir = os.path.dirname(input_file)
        # Unsilence needs to run from the input file's directory
        with changed_working_directory(input_dir):
            u = Unsilence(os.path.basename(input_file))
            u.detect_silence()
            # Render to the target output path (relative to the changed dir)
            # We need the absolute path for the output file relative to the original dir
            absolute_output_path = os.path.abspath(output_file)
            u.render_media(
                absolute_output_path, # Use absolute path here
                audible_speed=audible_speed,
                silent_speed=silent_speed,
            )
        logging.info(f"Silence adjusted (sped up) in audio and saved to {output_file}")
        return {"status": "success", "message": ""}
    except Exception as e:
        logging.exception(f"Error during silence adjustment: {str(e)}")
        return {"status": "error", "message": str(e)}


def extract_audio(video_path, output_path):
    """
    Extracts the audio track from a video file using MoviePy.

    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path where the extracted audio file will be saved.
                           The format is determined by MoviePy based on the
                           extension, but explicitly set to Opus codec at 48kHz here.

    Returns:
        None: The function performs the extraction and logs the result.
              It doesn't explicitly return status but relies on MoviePy's
              exception handling for errors.

    Raises:
        Exception: Can raise various exceptions from MoviePy (e.g., related to
                   file reading, writing, codec issues) if the extraction fails.
                   These are not explicitly caught here but will propagate up.
    """
    logging.info(f"Extracting audio from video: {video_path}")

    try:
        video = VideoFileClip(video_path)  # Create VideoFileClip instance here
        # Extract audio using Opus codec and 48kHz sample rate
        video.audio.write_audiofile(output_path, codec="aac", fps=48000)
        video.close() # Close the video file handle
        logging.info(f"Audio extracted and saved to {output_path}")
    except Exception as e:
        logging.error(f"Error extracting audio from {video_path}: {e}")
        # Optionally re-raise or handle specific exceptions
        raise # Re-raise the exception after logging


