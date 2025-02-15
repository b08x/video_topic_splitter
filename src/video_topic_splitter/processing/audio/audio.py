# processing/audio/audio.py
#!/usr/bin/env python3

"""Audio processing utilities."""

import logging
import os
import subprocess
from contextlib import contextmanager

import ffmpeg
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from unsilence import Unsilence

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def convert_to_mono_and_resample(input_file, output_file, sample_rate=16000):
    """Converts audio to mono, resamples, applies gain control, and a high-pass filter."""
    try:
        command = [
            "ffmpeg",
            "-i",
            input_file,
            "-af",
            "volume=-3dB,highpass=f=200, acompressor=threshold=-20dB:ratio=2:attack=5:release=50",
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
    """Normalizes audio using ffmpeg-normalize, then re-encodes to AAC."""
    output_ext = os.path.splitext(output_file)[1].lower()

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
            f"volume=-3dB,highpass=f={highpass_freq}",
            "-prf",
            "dynaudnorm=p=0.4:s=15",
            "-pof",
            f"lowpass=f={lowpass_freq}",
            "-ar",
            "48000",
        ]

        # Add codec settings based on output format
        if output_ext in [".mp4", ".m4a"]:
            command.extend(
                [
                    "-c:a",
                    "libopus",
                ]
            )
        else:
            command.extend(
                [
                    "-c:a",
                    "pcm_s16le",
                ]
            )

        command.extend(
            [
                "--keep-loudness-range-target",
                "-o",
                output_file,
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
    """Context manager to temporarily change the working directory."""
    old_dir = os.getcwd()
    try:
        os.chdir(new_dir)
        yield
    finally:
        os.chdir(old_dir)


def remove_silence(input_file, output_file, audible_speed=2, silent_speed=8):
    """Removes silence from audio using the unsilence library, creating an output directory."""
    try:
        # Get the directory for the output file (creating parent dirs if needed)
        output_dir = os.path.dirname(output_file)
        os.makedirs(
            output_dir, exist_ok=True
        )  # Create output directory if it doesn't exist

        input_dir = os.path.dirname(input_file)
        with changed_working_directory(input_dir):
            u = Unsilence(os.path.basename(input_file))
            u.detect_silence()
            u.render_media(
                os.path.basename(output_file),
                audible_speed=audible_speed,
                silent_speed=silent_speed,
            )
        logging.info(f"Silence adjusted (sped up) in audio and saved to {output_file}")
        return {"status": "success", "message": ""}
    except Exception as e:
        logging.exception(f"Error during silence adjustment: {str(e)}")
        return {"status": "error", "message": str(e)}


def extract_audio(video_path, output_path):
    """Extract audio from video file using ffmpeg-python, specifying AAC codec."""
    logging.info("Extracting audio from video...")

    video = VideoFileClip(video_path)  # Create VideoFileClip instance here

    video.audio.write_audiofile(output_path, codec="opus", fps=48000)

    logging.info(f"Audio extracted and saved to {output_path}")
