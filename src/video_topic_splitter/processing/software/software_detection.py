#!/usr/bin/env python3
"""Software detection functionality, primarily focusing on logo detection using template matching."""

import logging
import os

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def detect_software_logos(frame, software_list=None, logo_db_path=None, threshold=0.8):
    """
    Analyzes an image frame to detect logos of specified software using template matching.

    Compares pre-defined logo image templates (e.g., 'photoshop.png') found in
    `logo_db_path` against the input frame. It searches for logos corresponding
    to the names provided in `software_list`.

    Args:
        frame (np.ndarray): The input image frame (BGR format from OpenCV).
        software_list (list[str], optional): A list of software names whose logos
                                             should be searched for. If None or empty,
                                             the function returns an empty list.
                                             Defaults to None.
        logo_db_path (str, optional): The file path to the directory containing
                                      the logo template images. Logo files should be
                                      named like `softwarename.png` (lowercase).
                                      If None or path doesn't exist, returns empty list.
                                      Defaults to None.
        threshold (float, optional): The confidence threshold (0.0 to 1.0) for
                                     template matching (using TM_CCOEFF_NORMED).
                                     Only matches with a score greater than or equal
                                     to this threshold are considered valid.
                                     Defaults to 0.8.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary represents a
                    detected logo instance and contains:
                    - 'software' (str): The name of the detected software.
                    - 'confidence' (float): The matching score (0.0 to 1.0).
                    - 'location' (dict): A dictionary with 'x' and 'y' keys
                                         representing the top-left corner coordinates
                                         of the matched logo in the frame.
                    Returns an empty list if no logos are detected above the threshold
                    or if input parameters are invalid.
    """
    results = []

    # Basic input validation
    if not software_list or not logo_db_path or not os.path.isdir(logo_db_path):
        if not software_list:
            logger.warning("No software list provided for logo detection.")
        if not logo_db_path:
            logger.warning("No logo database path provided.")
        elif not os.path.isdir(logo_db_path):
            logger.warning(f"Logo database path not found or not a directory: {logo_db_path}")
        return results

    # Convert frame to grayscale once for efficiency
    try:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except cv2.error as e:
        logger.error(f"Error converting frame to grayscale: {e}")
        return results # Cannot proceed without grayscale frame

    for software in software_list:
        # Construct expected logo file path (lowercase name)
        logo_filename = f"{software.lower()}.png" # Assuming PNG format
        logo_path = os.path.join(logo_db_path, logo_filename)

        if not os.path.exists(logo_path):
            logger.debug(f"Logo template not found for {software} at {logo_path}")
            continue # Skip if logo file doesn't exist

        try:
            # Read the template logo image
            template = cv2.imread(logo_path, cv2.IMREAD_COLOR) # Read as color first
            if template is None:
                logger.warning(f"Could not read logo template for {software} from {logo_path}")
                continue

            # Check if template is smaller than frame
            if template.shape[0] > frame.shape[0] or template.shape[1] > frame.shape[1]:
                logger.warning(f"Logo template for {software} is larger than the frame. Skipping.")
                continue

            # Convert template to grayscale
            gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            template_h, template_w = gray_template.shape[:2]

            # Perform template matching
            # TM_CCOEFF_NORMED is generally robust for varying lighting
            match_result = cv2.matchTemplate(gray_frame, gray_template, cv2.TM_CCOEFF_NORMED)

            # Find locations where the match score exceeds the threshold
            locations = np.where(match_result >= threshold)

            # Process detected locations
            # zip(*locations[::-1]) converts row, col pairs to x, y pairs
            for pt in zip(*locations[::-1]):
                confidence_score = float(match_result[pt[1], pt[0]]) # Get score at this point
                # Store result with top-left corner (pt) and dimensions (w, h) implicitly known
                results.append(
                    {
                        "software": software,
                        "confidence": confidence_score,
                        "location": {
                            "x": int(pt[0]),
                            "y": int(pt[1]),
                            "w": template_w, # Add width
                            "h": template_h  # Add height for bounding box info
                            },
                    }
                )
                # Optional: Add Non-Maximum Suppression (NMS) here if multiple
                # overlapping boxes for the same logo are detected.

        except cv2.error as e:
            logger.error(f"OpenCV error matching logo for {software}: {str(e)}")
            continue # Continue with the next software
        except Exception as e:
            logger.error(f"Unexpected error matching logo for {software}: {str(e)}")
            continue # Continue with the next software

    if results:
        logger.info(f"Detected {len(results)} logo instances.")
    else:
        logger.info("No software logos detected meeting the criteria.")

    return results
