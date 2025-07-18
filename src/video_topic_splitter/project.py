#!/usr/bin/env python3
"""
Project management functionality for creating and managing processing folders.

This module handles the setup of project directories, ensuring that each video
or input source has a unique, organized space for its output files. It also
manages saving and loading processing checkpoints to allow for resumption of
interrupted workflows.
"""

import glob
import os
import pickle
import time
from typing import Any, Dict, Optional

from .constants import CHECKPOINTS
from .utils.youtube import is_youtube_url


def create_project_folder(input_path: str, base_output_dir: str) -> str:
    """
    Create or find a project folder for processing.

    If a project folder for the given input (video file or YouTube URL)
    already exists, the most recent one is used. Otherwise, a new folder is
    created with a timestamp. The folder name is derived from the input's
    base name.

    Args:
        input_path: The path to the input video file or a YouTube URL.
        base_output_dir: The base directory where project folders will be
            created.

    Returns:
        The absolute path to the created or found project folder.
    """
    if is_youtube_url(input_path):
        # Extract video ID from YouTube URL
        if "youtu.be/" in input_path:
            base_name = input_path.split("youtu.be/")[-1]
        else:
            base_name = input_path.split("v=")[-1].split("&")[0]
        base_name = f"yt_{base_name}"  # Prefix with 'yt_' to identify YouTube videos
    else:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
    project_pattern = os.path.join(base_output_dir, f"{base_name}_*")
    existing_projects = glob.glob(project_pattern)

    if existing_projects:
        # Use the most recent project folder
        project_path = max(existing_projects, key=os.path.getctime)
        print(f"Using existing project folder: {project_path}")
    else:
        # Create a new project folder
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        project_name = f"{base_name}_{timestamp}"
        project_path = os.path.join(base_output_dir, project_name)
        os.makedirs(project_path, exist_ok=True)
        print(f"Created new project folder: {project_path}")

    save_checkpoint(
        project_path, CHECKPOINTS["PROJECT_CREATED"], {"project_path": project_path}
    )
    return project_path


def save_checkpoint(project_path: str, stage: int, data: Dict[str, Any]) -> None:
    """
    Save the current processing stage and data to a checkpoint file.

    Uses pickle to serialize a dictionary containing the processing stage
    and any relevant data needed for resumption.

    Args:
        project_path: The path to the project folder.
        stage: An integer representing the last completed processing stage,
               as defined in `constants.CHECKPOINTS`.
        data: A dictionary containing data to be saved (e.g., file paths,
              intermediate results).
    """
    checkpoint_file = os.path.join(project_path, "checkpoint.pkl")
    with open(checkpoint_file, "wb") as f:
        pickle.dump({"stage": stage, "data": data}, f)


def load_checkpoint(project_path: str) -> Optional[Dict[str, Any]]:
    """
    Load a processing checkpoint from a file.

    If a checkpoint file exists, it is deserialized using pickle.

    Args:
        project_path: The path to the project folder.

    Returns:
        A dictionary containing the saved 'stage' and 'data', or None if
        the checkpoint file does not exist.
    """
    checkpoint_file = os.path.join(project_path, "checkpoint.pkl")
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "rb") as f:
            return pickle.load(f)
    return None
