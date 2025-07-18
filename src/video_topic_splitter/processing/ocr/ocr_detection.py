# processing/ocr/ocr_detection.py
#!/usr/bin/env python3
"""OCR functionality for detecting software application text in video frames."""

import logging

import cv2
import numpy as np
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)


def preprocess_frame(frame):
    """
    Preprocess a video frame to improve OCR accuracy.

    This function applies a series of image processing techniques, including
    converting the frame to grayscale, applying a binary threshold to create a
    black-and-white image, and removing noise.

    Args:
        frame: The input video frame as a NumPy array.

    Returns:
        The preprocessed frame as a NumPy array.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to get black text on white background
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Noise removal
    denoised = cv2.fastNlMeansDenoising(binary)

    return denoised


def detect_text_regions(frame):
    """
    Detect potential regions of text within a video frame.

    This uses the MSER (Maximally Stable Extremal Regions) algorithm to find
    areas in the frame that are likely to contain text, and filters them based
    on their size and aspect ratio.

    Args:
        frame: The input video frame as a NumPy array.

    Returns:
        A list of tuples, where each tuple represents the bounding box
        (x, y, w, h) of a potential text region.
    """
    # Convert to grayscale if not already
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    # Apply MSER (Maximally Stable Extremal Regions)
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)

    # Convert regions to rectangles
    text_regions = []
    for region in regions:
        x, y, w, h = cv2.boundingRect(region)
        # Filter regions based on aspect ratio and size
        aspect_ratio = w / float(h)
        if 0.1 < aspect_ratio < 10 and w > 10 and h > 5:
            text_regions.append((x, y, w, h))

    return text_regions


def perform_ocr(frame, regions=None, lang="eng"):
    """
    Perform OCR on a frame, either on the entire frame or on specific regions.

    This function uses the Tesseract OCR engine to extract text from the given
    frame. If regions are provided, it performs OCR on each region
    individually for better accuracy.

    Args:
        frame: The input frame, either as a NumPy array or a PIL Image.
        regions: An optional list of bounding boxes for text regions.
        lang: The language to be used for OCR.

    Returns:
        A list of dictionaries, where each dictionary contains the detected
        text, its bounding box, and a confidence score.
    """
    try:
        # Convert frame to PIL Image for Tesseract
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        results = []

        if regions:
            # Process each region separately
            for x, y, w, h in regions:
                region = frame.crop((x, y, x + w, y + h))
                text = pytesseract.image_to_string(region, lang=lang)
                if text.strip():
                    results.append(
                        {
                            "text": text.strip(),
                            "bbox": (x, y, w, h),
                            "confidence": pytesseract.image_to_data(
                                region, lang=lang, output_type=pytesseract.Output.DICT
                            )["conf"][0],
                        }
                    )
        else:
            # Process entire frame
            text = pytesseract.image_to_string(frame, lang=lang)
            if text.strip():
                results.append(
                    {
                        "text": text.strip(),
                        "bbox": None,
                        "confidence": pytesseract.image_to_data(
                            frame, lang=lang, output_type=pytesseract.Output.DICT
                        )["conf"][0],
                    }
                )

        return results

    except Exception as e:
        logger.error(f"OCR processing error: {str(e)}")
        return []


def detect_software_names(frame, software_list, lang="eng"):
    """
    Detect specified software application names within a video frame.

    This function orchestrates the entire OCR process: it preprocesses the
    frame, detects text regions, performs OCR, and then matches the detected
    text against a provided list of software names.

    Args:
        frame: The input video frame as a NumPy array.
        software_list: A list of software names to search for.
        lang: The language to be used for OCR.

    Returns:
        A list of dictionaries, where each dictionary represents a detected
        software name and includes details about the match.
    """
    # Preprocess the frame
    processed_frame = preprocess_frame(frame)

    # Detect text regions
    regions = detect_text_regions(processed_frame)

    # Perform OCR
    ocr_results = perform_ocr(processed_frame, regions, lang=lang)

    # Match detected text against software list
    matches = []
    for result in ocr_results:
        text = result["text"].lower()
        for software in software_list:
            if software.lower() in text:
                matches.append(
                    {
                        "software": software,
                        "detected_text": result["text"],
                        "confidence": result["confidence"],
                        "location": result["bbox"],
                    }
                )

    return matches
