#!/usr/bin/env python3
"""Logo detection functionality for identifying software applications in video frames."""

import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class LogoDetector:
    """Class for managing and performing logo detection."""

    def __init__(self, logo_db_path=None):
        """Initialize the logo detector with a database of reference logos."""
        self.logo_db_path = logo_db_path or os.path.join(
            os.path.dirname(__file__), "data", "logos"
        )
        self.logos = {}
        self.load_logo_database()

    def load_logo_database(self):
        """Load reference logos from the database directory."""
        try:
            if not os.path.exists(self.logo_db_path):
                os.makedirs(self.logo_db_path)

            # Load logo metadata if exists
            metadata_path = os.path.join(self.logo_db_path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    self.logos = json.load(f)

            # Load logo images
            for logo_file in Path(self.logo_db_path).glob("*.png"):
                software_name = logo_file.stem
                if software_name not in self.logos:
                    self.logos[software_name] = {
                        "path": str(logo_file),
                        "template": cv2.imread(str(logo_file), cv2.IMREAD_UNCHANGED),
                    }

        except Exception as e:
            logger.error(f"Error loading logo database: {str(e)}")
            self.logos = {}

    def add_logo(self, software_name, logo_image):
        """Add a new logo to the database."""
        try:
            # Ensure the logo directory exists
            os.makedirs(self.logo_db_path, exist_ok=True)

            # Save the logo image
            logo_path = os.path.join(self.logo_db_path, f"{software_name}.png")
            cv2.imwrite(logo_path, logo_image)

            # Add to loaded logos
            self.logos[software_name] = {"path": logo_path, "template": logo_image}

            # Update metadata
            self._save_metadata()

            return True
        except Exception as e:
            logger.error(f"Error adding logo: {str(e)}")
            return False

    def _save_metadata(self):
        """Save logo database metadata."""
        try:
            metadata = {
                name: {"path": info["path"]} for name, info in self.logos.items()
            }
            metadata_path = os.path.join(self.logo_db_path, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")

    def detect_logos(self, frame, threshold=0.8):
        """Detect software logos in a video frame."""
        results = []
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for software_name, logo_info in self.logos.items():
            template = logo_info["template"]

            # Convert template to grayscale if it's not already
            if len(template.shape) > 2:
                template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

            # Perform template matching
            try:
                result = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
                locations = np.where(result >= threshold)

                for pt in zip(*locations[::-1]):  # Switch columns and rows
                    results.append(
                        {
                            "software": software_name,
                            "confidence": float(result[pt[1], pt[0]]),
                            "location": {
                                "x": int(pt[0]),
                                "y": int(pt[1]),
                                "width": template.shape[1],
                                "height": template.shape[0],
                            },
                        }
                    )
            except Exception as e:
                logger.error(f"Error matching template for {software_name}: {str(e)}")
                continue

        return results


def detect_software_logos(frame, software_list, logo_db_path=None, threshold=0.8):
    """Convenience function to detect software logos in a frame.

    Args:
        frame: Video frame to analyze
        software_list: List of software names to detect
        logo_db_path: Optional path to logo database
        threshold: Confidence threshold for logo detection (0.0-1.0, default: 0.8)
    """
    detector = LogoDetector(logo_db_path)
    return detector.detect_logos(frame, threshold=threshold)
