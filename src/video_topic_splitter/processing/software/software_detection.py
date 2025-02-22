#!/usr/bin/env python3
"""Software detection functionality including logo detection."""

import logging
import os

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def detect_software_logos(frame, software_list=None, logo_db_path=None, threshold=0.8):
    """Analyze a frame for software logos using template matching.

    Args:
        frame: OpenCV image frame to analyze
        software_list: Optional list of software names to detect
        logo_db_path: Optional path to logo database directory
        threshold: Confidence threshold for matches (0-1)

    Returns:
        List of dictionaries containing detected software logos with confidence scores
        and locations
    """
    results = []

    if not software_list or not logo_db_path or not os.path.exists(logo_db_path):
        return results

    for software in software_list:
        # Look for logo template files
        logo_path = os.path.join(logo_db_path, f"{software.lower()}.png")
        if not os.path.exists(logo_path):
            continue

        try:
            # Read and match template
            template = cv2.imread(logo_path)
            if template is None:
                continue

            # Convert both to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

            # Template matching
            result = cv2.matchTemplate(gray_frame, gray_template, cv2.TM_CCOEFF_NORMED)

            # Get matches above threshold
            locations = np.where(result >= threshold)
            for pt in zip(*locations[::-1]):  # Switch columns and rows
                results.append(
                    {
                        "software": software,
                        "confidence": float(result[pt[1]][pt[0]]),
                        "location": {"x": int(pt[0]), "y": int(pt[1])},
                    }
                )

        except Exception as e:
            logger.error(f"Error matching logo for {software}: {str(e)}")
            continue

    return results
