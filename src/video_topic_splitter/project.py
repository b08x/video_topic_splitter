#!/usr/bin/env python3
"""Project management functionality."""

import glob
import os
import pickle
import time

from .constants import CHECKPOINTS
from .utils.youtube import is_youtube_url


def create_project_folder(input_path, base_output_dir):
    """Create or find project folder for processing."""
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


def save_checkpoint(project_path, stage, data):
    """Save processing checkpoint."""
    checkpoint_file = os.path.join(project_path, "checkpoint.pkl")
    with open(checkpoint_file, "wb") as f:
        pickle.dump({"stage": stage, "data": data}, f)


def load_checkpoint(project_path):
    """Load processing checkpoint."""
    checkpoint_file = os.path.join(project_path, "checkpoint.pkl")
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "rb") as f:
            return
