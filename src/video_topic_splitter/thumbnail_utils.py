#!/usr/bin/env python3
"""Thumbnail generation and management utilities."""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from PIL import Image

logger = logging.getLogger(__name__)


class ThumbnailManager:
    """Manages thumbnail generation, storage, and analysis.

    Attributes:
        project_path (str): The root path of the project.
        thumbnails_dir (str): The directory to store generated thumbnails.
        metadata_path (str): Path to the JSON file storing thumbnail metadata.
        metadata (dict): A dictionary containing thumbnail metadata.
    """

    def __init__(self, project_path: str):
        """Initialize the thumbnail manager.

        Args:
            project_path (str): The root path of the project.  Thumbnails will be stored in a subdirectory.
        """
        self.project_path = project_path
        self.thumbnails_dir = os.path.join(project_path, "thumbnails")
        self.metadata_path = os.path.join(self.thumbnails_dir, "metadata.json")
        self.metadata = self._load_metadata()

        # Create thumbnails directory if it doesn't exist
        os.makedirs(self.thumbnails_dir, exist_ok=True)

    def _load_metadata(self) -> Dict:
        """Load thumbnail metadata from JSON file.

        Returns:
            dict: A dictionary containing thumbnail metadata. Returns an empty dictionary if the file doesn't exist or if loading fails.
        """
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading thumbnail metadata: {str(e)}")
        return {"thumbnails": [], "last_updated": datetime.now().isoformat()}

    def _save_metadata(self):
        """Save thumbnail metadata to JSON file.

        Updates the 'last_updated' timestamp and saves the metadata to the JSON file. Handles potential errors during saving.
        """
        try:
            self.metadata["last_updated"] = datetime.now().isoformat()
            with open(self.metadata_path, "w") as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving thumbnail metadata: {str(e)}")

    def generate_thumbnails(
        self, video_path: str, interval: int = 30, max_thumbnails: int = 10
    ) -> List[Dict]:
        """Generate thumbnails from a video file at specified intervals.

        Args:
            video_path (str): Path to the video file.
            interval (int, optional): Interval in seconds between thumbnails. Defaults to 30.
            max_thumbnails (int, optional): Maximum number of thumbnails to generate. Defaults to 10.

        Returns:
            list: A list of dictionaries, where each dictionary contains information about a generated thumbnail (path, time, index). Returns an empty list if thumbnail generation fails.
        """
        try:
            video = VideoFileClip(video_path)
            duration = video.duration

            # Calculate number of thumbnails based on interval and max_thumbnails
            num_thumbnails = min(int(duration / interval), max_thumbnails)
            if num_thumbnails == 0:
                num_thumbnails = 1  # At least one thumbnail

            # Calculate actual interval to spread thumbnails evenly
            actual_interval = duration / num_thumbnails

            thumbnails = []
            for i in range(num_thumbnails):
                time = i * actual_interval
                # Get frame at specified time
                frame = video.get_frame(time)

                # Convert to PIL Image
                image = Image.fromarray(frame)

                # Save thumbnail
                thumbnail_path = os.path.join(
                    self.thumbnails_dir, f"thumbnail_{i:03d}.jpg"
                )
                image.save(thumbnail_path, "JPEG", quality=85)

                thumbnail_info = {"path": thumbnail_path, "time": time, "index": i}
                thumbnails.append(thumbnail_info)

            video.close()

            # Update metadata
            self.metadata["thumbnails"] = thumbnails
            self._save_metadata()

            return thumbnails

        except Exception as e:
            logger.error(f"Error generating thumbnails: {str(e)}")
            return []

    def save_youtube_thumbnail(
        self, thumbnail_url: str, index: int = 0
    ) -> Optional[Dict]:
        """Save a YouTube thumbnail from URL.

        Args:
            thumbnail_url (str): URL of the YouTube thumbnail.
            index (int, optional): Index for the thumbnail. Defaults to 0.

        Returns:
            Optional[Dict]: A dictionary containing information about the saved thumbnail (path, url, index, source). Returns None if saving fails.
        """
        try:
            import requests

            # Download thumbnail
            response = requests.get(thumbnail_url)
            if response.status_code != 200:
                raise Exception(f"Failed to download thumbnail: {response.status_code}")

            # Save thumbnail
            thumbnail_path = os.path.join(
                self.thumbnails_dir, f"yt_thumbnail_{index:03d}.jpg"
            )
            with open(thumbnail_path, "wb") as f:
                f.write(response.content)

            thumbnail_info = {
                "path": thumbnail_path,
                "url": thumbnail_url,
                "index": index,
                "source": "youtube",
            }

            # Update metadata
            if "thumbnails" not in self.metadata:
                self.metadata["thumbnails"] = []
            self.metadata["thumbnails"].append(thumbnail_info)
            self._save_metadata()

            return thumbnail_info

        except Exception as e:
            logger.error(f"Error saving YouTube thumbnail: {str(e)}")
            return None

    def get_thumbnails(self) -> List[Dict]:
        """Get list of all thumbnails.

        Returns:
            list: A list of dictionaries, where each dictionary contains information about a thumbnail. Returns an empty list if no thumbnails are found.
        """
        return self.metadata.get("thumbnails", [])

    def clear_thumbnails(self):
        """Remove all thumbnails and reset metadata.

        Deletes all thumbnail files from the thumbnails directory and resets the metadata to an empty state. Handles potential errors during file deletion.
        """
        try:
            # Remove thumbnail files
            for thumbnail in self.get_thumbnails():
                try:
                    os.remove(thumbnail["path"])
                except Exception as e:
                    logger.error(
                        f"Error removing thumbnail {thumbnail['path']}: {str(e)}"
                    )

            # Reset metadata
            self.metadata = {
                "thumbnails": [],
                "last_updated": datetime.now().isoformat(),
            }
            self._save_metadata()

        except Exception as e:
            logger.error(f"Error clearing thumbnails: {str(e)}")

    def load_thumbnail(self, thumbnail_info: Dict) -> Optional[np.ndarray]:
        """Load a thumbnail as a numpy array.

        Args:
            thumbnail_info (dict): A dictionary containing thumbnail information, including the path.

        Returns:
            Optional[np.ndarray]: A NumPy array representing the loaded thumbnail image. Returns None if loading fails.
        """
        try:
            return cv2.imread(thumbnail_info["path"])
        except Exception as e:
            logger.error(f"Error loading thumbnail {thumbnail_info['path']}: {str(e)}")
            return None
