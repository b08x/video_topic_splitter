# utils/utils.py
#!/usr/bin/env python3
"""
General utility functions.
"""

import logging
import os
import subprocess
from typing import List, Optional, Tuple  # Added Tuple import

logger = logging.getLogger(__name__)


def run_command(command: List[str], check: bool = True) -> Optional[str]:
    """
    Executes a shell command and returns the output.

    Args:
        command: A list of strings representing the command and its arguments.
        check: If True, raises a CalledProcessError if the command exits with a
               non-zero exit code.

    Returns:
        The standard output of the command as a string, or None if an error occurred.
    """
    try:
        result = subprocess.run(
            command, check=check, capture_output=True, text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"Command '{' '.join(command)}' failed: {e.stderr}")
        return None
    except FileNotFoundError:
        logger.error(f"Command not found: {command[0]}")
        return None
    except Exception as e:
        logger.error(f"Error running command '{' '.join(command)}': {e}")
        return None


def ensure_directory_exists(dir_path: str) -> bool:
    """
    Ensures that a directory exists. Creates the directory if it doesn't exist.

    Args:
        dir_path: The path to the directory.

    Returns:
        True if the directory exists (either already or after creation), False otherwise.
    """
    try:
        # Create if not exists, no error if exists
        os.makedirs(dir_path, exist_ok=True)
        return True
    except OSError as e:
        logger.error(f"Failed to create directory '{dir_path}': {e}")
        return False


def split_video_ffmpeg(
    input_video_path: str,
    scene_list: List[Tuple[float, float]],
    output_file_template: str,
    show_progress: bool = False,  # Added default value
    show_output: bool = False,
    ffmpeg_args: Optional[List[str]] = None,
) -> List[str]:
    """
    Splits a video into multiple clips using FFmpeg.

    This function is a duplicate of the one in processing/video/scene_detection.py.
    It's included here for convenience and to avoid circular dependencies.
    Consider refactoring to avoid this duplication.

    Args:
        input_video_path: Path to the input video file.
        scene_list: A list of tuples, where each tuple represents the start and end
                    time (in seconds) of a scene.
        output_file_template: Template for naming the output video files.
                              "$SCENE_NUMBER" will be replaced with the scene index (1-based).
        show_progress: If True, displays FFmpeg's progress. (Note: FFmpeg progress display
                       might require specific handling not fully implemented here).
        show_output: If True, shows FFmpeg output.
        ffmpeg_args: Optional list of additional FFmpeg arguments.

    Returns:
        A list of paths to the generated video segment files.
    """
    if not scene_list:
        logger.warning(
            "No scenes provided for splitting. Returning empty list.")
        return []

    output_files: List[str] = []
    try:
        # Determine padding for scene numbers in filenames
        scene_count = len(scene_list)
        # Minimum 3 digits, or more if needed
        padding = max(3, len(str(scene_count)))

        for i, (start_time, end_time) in enumerate(scene_list):
            # 1-based scene number with padding
            scene_number = str(i + 1).zfill(padding)
            output_path = output_file_template.replace(
                "$SCENE_NUMBER", scene_number)
            output_path = os.path.abspath(output_path)  # Ensure absolute path

            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir:
                ensure_directory_exists(output_dir)

            # Construct FFmpeg command
            command = [
                "ffmpeg",
                "-ss", str(start_time),
                "-i", input_video_path,
                "-to", str(end_time),
                "-y",  # Overwrite output files without asking
            ]

            if ffmpeg_args:
                command.extend(ffmpeg_args)  # Add any extra FFmpeg arguments
            else:
                # Default: stream copy for speed, but can be less accurate with seeking
                # Consider adding '-copyts' to preserve timestamps if needed
                command.extend(["-c", "copy"])

            command.append(output_path)

            # Execute FFmpeg command
            logger.debug(f"Running FFmpeg command: {' '.join(command)}")
            process = subprocess.run(
                command,
                check=True,
                capture_output=not show_output,  # Capture if not showing output
                text=True,
                # Use stderr=subprocess.PIPE if you want to capture errors even when show_output=True
                stderr=subprocess.PIPE if not show_output else None
            )

            # Check if the file was created successfully
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                output_files.append(output_path)
                logger.info(f"Segment saved to {output_path}")
            else:
                logger.warning(
                    f"FFmpeg command executed but output file is missing or empty: {output_path}")
                # Log stderr if captured
                if process.stderr:
                    logger.warning(f"FFmpeg stderr: {process.stderr.strip()}")

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error executing command: {' '.join(e.cmd)}")
        # Log stderr if available in the exception
        stderr_output = e.stderr.strip() if e.stderr else "No stderr captured."
        logger.error(f"FFmpeg stderr: {stderr_output}")
    except FileNotFoundError:
        logger.error(
            "FFmpeg command not found. Ensure FFmpeg is installed and in your PATH.")
    except Exception as e:
        logger.error(
            f"Unexpected error during video splitting: {e}", exc_info=True)

    # Verify the number of output files matches the expected count
    if len(output_files) != len(scene_list):
        logger.warning(
            f"Expected {len(scene_list)} segments, but only {len(output_files)} were successfully created.")

    return output_files
