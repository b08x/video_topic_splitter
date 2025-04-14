#!/usr/bin/env python3
"""Command-line interface utilities for video topic splitting projects.

This module provides core functionality for managing the lifecycle of video
processing projects. It handles the creation and identification of project-specific
directories where all intermediate files, results, and state checkpoints are
stored. This ensures that processing for different videos is isolated and
that long-running tasks can be potentially resumed.

Key features include:
- Generating unique project folder names based on input video file paths or
  YouTube URLs.
- Identifying existing project folders to potentially reuse or resume work.
- Saving and loading processing state (checkpoints) using Python's pickle
  module, allowing tasks to pick up where they left off.
"""

import glob
import os
import pickle
import time
from typing import Any, Dict, Optional
import logging # Added for potential logging improvements
import re # Added for more robust URL parsing

from .constants import CHECKPOINTS
from .utils.youtube import is_youtube_url # Assuming this utility exists and works correctly

# Consider adding basic logging configuration if not handled elsewhere
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _extract_youtube_video_id(url: str) -> Optional[str]:
    """Extracts the YouTube video ID from various URL formats.

    Supports standard watch URLs, short youtu.be URLs, and embed URLs.

    Args:
        url: The YouTube URL string.

    Returns:
        The extracted video ID string, or None if no valid ID is found.
    """
    # Regex patterns to cover common YouTube URL formats
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # Standard watch?v= or /v/
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})', # Short youtu.be/
        r'(?:embed\/)([0-9A-Za-z_-]{11})'    # Embed URL /embed/
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    logger.warning(f"Could not extract YouTube video ID from URL: {url}")
    return None


def create_project_folder(input_path: str, base_output_dir: str) -> str:
    """Creates or finds an existing project folder for a given input video or URL.

    This function determines the appropriate base name for the project folder.
    If the input is a recognized YouTube URL, the video ID is extracted and
    used as the base name, prefixed with 'yt_' (e.g., 'yt_VIDEOID').
    If the input is a local file path, the base name of the file (without its
    extension) is used (e.g., 'my_video').

    The function then searches within the `base_output_dir` for existing folders
    matching the pattern '{base_name}_*'. If one or more matching folders are
    found, it selects the one with the most recent creation timestamp (using
    `os.path.getctime`) and returns its absolute path. This allows reusing
    previous processing results or resuming interrupted tasks.

    If no existing project folder is found, a new one is created. The new
    folder's name follows the format '{base_name}_{timestamp}', where the
    timestamp is formatted as 'YYYYMMDD_HHMMSS'. The function ensures the
    directory is created using `os.makedirs` with `exist_ok=True`.

    Immediately after creating a new folder or identifying an existing one,
    an initial checkpoint (stage: `PROJECT_CREATED`) is saved within the
    project folder using `save_checkpoint`. This marks the successful
    initialization or loading of the project environment.

    Args:
        input_path: The path to the input video file or a valid YouTube URL string.
        base_output_dir: The path to the base directory where project folders
                         should be created or searched for. This directory
                         must exist.

    Returns:
        The absolute path to the created or found project folder.

    Raises:
        OSError: If there is an issue creating the new project directory (e.g.,
                 permission errors, invalid path).
        ValueError: If a YouTube URL is provided but a video ID cannot be
                    extracted.
    """
    if is_youtube_url(input_path):
        video_id = _extract_youtube_video_id(input_path)
        if not video_id:
            raise ValueError(f"Could not extract video ID from YouTube URL: {input_path}")
        base_name = f"yt_{video_id}"
        logger.info(f"Identified YouTube input. Using base name: {base_name}")
    else:
        if not os.path.isfile(input_path):
             # Add check if it's a file path but doesn't exist
             # Depending on requirements, you might raise an error or just log a warning
             logger.warning(f"Input path is not a YouTube URL and file does not exist: {input_path}")
             # Fallback to using the basename anyway, or raise FileNotFoundError
             # raise FileNotFoundError(f"Input file not found: {input_path}")
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        logger.info(f"Identified file input. Using base name: {base_name}")

    # Ensure base_output_dir exists before searching/creating projects within it
    if not os.path.isdir(base_output_dir):
        # Or raise an error if it *must* exist beforehand
        logger.info(f"Base output directory does not exist, creating: {base_output_dir}")
        try:
            os.makedirs(base_output_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create base output directory {base_output_dir}: {e}")
            raise # Re-raise the exception

    project_pattern = os.path.join(base_output_dir, f"{base_name}_*")
    logger.debug(f"Searching for existing projects with pattern: {project_pattern}")
    existing_projects = glob.glob(project_pattern)

    project_path: Optional[str] = None
    if existing_projects:
        # Filter out potential files that might match the glob pattern
        existing_dirs = [p for p in existing_projects if os.path.isdir(p)]
        if existing_dirs:
            # Sort by creation time (most recent first)
            # Note: ctime might be 'last metadata change time' on some Unix systems.
            # mtime (modification time) might be more reliable if files *within* the dir change.
            # If timestamp in name is reliable, sorting by name might be safer.
            try:
                 project_path = max(existing_dirs, key=os.path.getctime)
                 logger.info(f"Found existing project folder(s). Using most recent: {project_path}")
            except ValueError:
                 # This case should ideally not happen if existing_dirs is not empty
                 logger.warning("Found matching entries but none were directories.")
                 project_path = None # Ensure project_path is None if max fails

    if project_path is None: # If no existing dir found or max failed
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        project_name = f"{base_name}_{timestamp}"
        project_path = os.path.join(base_output_dir, project_name)
        try:
            os.makedirs(project_path, exist_ok=True)
            logger.info(f"Created new project folder: {project_path}")
        except OSError as e:
            logger.error(f"Failed to create project directory {project_path}: {e}")
            raise # Re-raise the exception

    # Ensure project_path is now definitely a string
    if project_path is None:
         # This should be unreachable if error handling above is correct, but acts as a safeguard
         raise RuntimeError("Failed to determine project path.")


    # Save an initial checkpoint indicating the project folder is ready
    # Use os.path.abspath to ensure consistency
    abs_project_path = os.path.abspath(project_path)
    save_checkpoint(
        abs_project_path, CHECKPOINTS["PROJECT_CREATED"], {"project_path": abs_project_path}
    )
    return abs_project_path


def save_checkpoint(project_path: str, stage: str, data: Dict[str, Any]) -> None:
    """Saves the current processing stage and associated data to a checkpoint file.

    This function serializes the provided `data` dictionary along with a `stage`
    identifier using Python's `pickle` module. The serialized data is written
    in binary format ('wb') to a file named 'checkpoint.pkl' located directly
    within the specified `project_path`.

    Checkpoints allow the application to store its state at various points
    during a potentially long-running process. If the process is interrupted,
    it can be potentially resumed later by loading this checkpoint file using
    `load_checkpoint`.

    Args:
        project_path: The absolute path to the project folder where the
                      'checkpoint.pkl' file should be saved.
        stage: A string identifier representing the completed processing stage
               (e.g., 'TRANSCRIPTION_COMPLETE'). It's recommended to use
               predefined constants, such as those from `constants.CHECKPOINTS`.
        data: A dictionary containing any data relevant to the completed stage
              that would be necessary or useful for resuming the process later.
              This could include file paths, intermediate results, configuration, etc.

    Raises:
        FileNotFoundError: If the `project_path` directory does not exist.
        OSError: If there are file system errors during file writing (e.g.,
                 permissions, disk full).
        pickle.PicklingError: If the provided `data` object cannot be serialized
                              by pickle.
    """
    if not os.path.isdir(project_path):
        # It's generally better to ensure the path exists before calling save
        logger.error(f"Project path does not exist, cannot save checkpoint: {project_path}")
        raise FileNotFoundError(f"Project directory not found: {project_path}")

    checkpoint_file = os.path.join(project_path, "checkpoint.pkl")
    logger.info(f"Saving checkpoint for stage '{stage}' to {checkpoint_file}")
    try:
        # Use 'wb' mode for writing binary data (pickle)
        with open(checkpoint_file, "wb") as f:
            checkpoint_data = {"stage": stage, "data": data}
            pickle.dump(checkpoint_data, f)
        logger.info(f"Checkpoint saved successfully for stage: {stage}")
    except (OSError, pickle.PicklingError) as e:
        logger.error(f"Error saving checkpoint file {checkpoint_file}: {e}")
        # Re-raise the exception to signal failure
        raise


def load_checkpoint(project_path: str) -> Optional[Dict[str, Any]]:
    """Loads the processing checkpoint data from the project folder.

    This function looks for a file named 'checkpoint.pkl' within the specified
    `project_path`. If the file exists, it attempts to open it in binary read
    mode ('rb') and deserialize its contents using `pickle.load`.

    If the file is found and successfully deserialized, the function returns
    the loaded data, which is expected to be a dictionary containing at least
    a 'stage' key indicating the last successfully completed stage, and a 'data'
    key holding the associated state information.

    If the 'checkpoint.pkl' file does not exist in the `project_path`, the
    function returns `None`.

    If the file exists but an error occurs during file reading or deserialization
    (e.g., the file is corrupted, empty, or was created with an incompatible
    pickle protocol or Python version), an error is logged, and the function
    returns `None`.

    Args:
        project_path: The absolute path to the project folder where the
                      'checkpoint.pkl' file is expected to be located.

    Returns:
        A dictionary containing the 'stage' and 'data' from the last saved
        checkpoint if the file exists and is loaded successfully. Returns `None`
        if the checkpoint file does not exist or if an error occurs during
        loading or deserialization.
    """
    checkpoint_file = os.path.join(project_path, "checkpoint.pkl")
    if os.path.exists(checkpoint_file):
        logger.info(f"Attempting to load checkpoint from: {checkpoint_file}")
        try:
            # Use 'rb' mode for reading binary data (pickle)
            with open(checkpoint_file, "rb") as f:
                checkpoint_data = pickle.load(f)
                if isinstance(checkpoint_data, dict) and 'stage' in checkpoint_data:
                    logger.info(
                        f"Checkpoint loaded. Last completed stage: {checkpoint_data.get('stage', 'Unknown')}"
                    )
                    return checkpoint_data
                else:
                    logger.warning(f"Checkpoint file {checkpoint_file} has unexpected format.")
                    return None # Or handle as corrupted
        except (pickle.UnpicklingError, EOFError, FileNotFoundError, OSError, TypeError) as e:
            # FileNotFoundError might occur in rare race conditions
            # OSError for general I/O errors
            # TypeError can occur with incompatible pickle data
            logger.error(f"Error loading checkpoint file {checkpoint_file}: {e}")
            # Optionally, handle corrupted/empty file (e.g., delete it, rename it)
            # os.rename(checkpoint_file, checkpoint_file + ".corrupted")
            return None
        except Exception as e:
             # Catch any other unexpected errors during loading
             logger.error(f"Unexpected error loading checkpoint {checkpoint_file}: {e}", exc_info=True)
             return None
    else:
        logger.info(f"No checkpoint file found at: {checkpoint_file}")
        return None

