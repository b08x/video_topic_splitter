#!/usr/bin/env python3
"""Project management functionality."""

import glob
import os
import pickle
import time

from .constants import CHECKPOINTS
from .youtube import is_youtube_url


def create_project_folder(input_path: str, base_output_dir: str) -> str:
    """Creates a new project folder or reuses an existing one.

    Args:
        input_path (str): Path to the input video file or YouTube URL.
        base_output_dir (str): Base directory for project folders.

    Returns:
        str: The path to the created or existing project folder.
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


def save_checkpoint(project_path: str, stage: int, data: dict):
    """Saves a processing checkpoint to a pickle file.

    Args:
        project_path (str): Path to the project directory.
        stage (int): The current processing stage (from constants.CHECKPOINTS).
        data (dict): Dictionary containing data to be saved.
    """
    checkpoint_file = os.path.join(project_path, "checkpoint.pkl")
    with open(checkpoint_file, "wb") as f:
        pickle.dump({"stage": stage, "data": data}, f)


def load_checkpoint(project_path: str) -> Optional[dict]:
    """Loads a processing checkpoint from a pickle file.

    Args:
        project_path (str): Path to the project directory.

    Returns:
        Optional[dict]: The loaded checkpoint data (dictionary) or None if no checkpoint file exists.
    """
    checkpoint_file = os.path.join(project_path, "checkpoint.pkl")
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "rb") as f:
            return pickle.load(f)
    return None
