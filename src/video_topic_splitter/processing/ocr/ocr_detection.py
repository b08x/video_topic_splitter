# processing/ocr/ocr_detection.py
#!/usr/bin/env python3
"""OCR functionality for detecting text, particularly software names, in video frames."""

import logging

import cv2
import numpy as np
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)


def preprocess_frame(frame):
    """
    Preprocesses an image frame to enhance text visibility for OCR.

    Steps include:
    1. Convert to grayscale.
    2. Apply Otsu's thresholding to create a binary image (black text on white background).
    3. Apply fast Non-Local Means Denoising to reduce noise.

    Args:
        frame (np.ndarray): The input image frame (BGR format from OpenCV).

    Returns:
        np.ndarray: The preprocessed image frame (grayscale, binary, denoised).
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to get black text on white background
    # THRESH_BINARY_INV might be better if text is light on dark background
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Noise removal - consider adjusting h parameter if needed
    denoised = cv2.fastNlMeansDenoising(binary, h=30) # Increased h slightly

    return denoised


def detect_text_regions(frame):
    """
    Detects potential text regions within an image frame using MSER.

    Maximally Stable Extremal Regions (MSER) is used to find connected
    components that are stable over intensity changes, often corresponding
    to text characters. Detected regions are filtered based on aspect ratio
    and size to reduce non-text noise.

    Args:
        frame (np.ndarray): The input image frame (preferably grayscale,
                            like the output of preprocess_frame).

    Returns:
        list[tuple[int, int, int, int]]: A list of bounding boxes (x, y, w, h)
                                         representing potential text regions.
    """
    # Convert to grayscale if not already
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    # Apply MSER (Maximally Stable Extremal Regions)
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)

    # Convert regions to bounding boxes and filter
    text_regions = []
    for region in regions:
        # Get bounding box for the contour points
        x, y, w, h = cv2.boundingRect(region)
        # Filter regions based on aspect ratio and size heuristics
        aspect_ratio = w / float(h) if h > 0 else 0 # Avoid division by zero
        # Adjust thresholds as needed based on expected text characteristics
        if 0.1 < aspect_ratio < 10 and w > 10 and h > 5 and w * h > 50:
            text_regions.append((x, y, w, h))

    # Optional: Add non-maximum suppression (NMS) here if regions overlap heavily

    return text_regions


def perform_ocr(frame, regions=None, lang="eng"):
    """
    Performs Optical Character Recognition (OCR) on an image or specific regions.

    Uses the Tesseract OCR engine via the pytesseract wrapper. If regions
    are provided, OCR is performed on each region individually. Otherwise,
    it's performed on the entire frame.

    Args:
        frame (np.ndarray | PIL.Image.Image): The input image frame (can be
                                               OpenCV BGR ndarray or PIL Image).
        regions (list[tuple[int, int, int, int]], optional): A list of
                                                             bounding boxes (x, y, w, h)
                                                             to perform OCR on. If None,
                                                             OCR is done on the whole frame.
                                                             Defaults to None.
        lang (str, optional): The language code for Tesseract (e.g., 'eng', 'fra').
                              Defaults to "eng".

    Returns:
        list[dict]: A list of dictionaries, each containing:
                    - 'text' (str): The detected text (stripped of whitespace).
                    - 'bbox' (tuple | None): The bounding box (x, y, w, h) of the
                                             region where text was found, or None
                                             if OCR was on the whole frame.
                    - 'confidence' (float): Tesseract's confidence score for the
                                            detected text block (often the first word's
                                            confidence, may not be highly reliable).
                    Returns an empty list if no text is found or an error occurs.
    """
    try:
        # Convert frame to PIL Image for Tesseract if it's an OpenCV ndarray
        if isinstance(frame, np.ndarray):
            # Assume BGR input from OpenCV, convert to RGB for PIL/Tesseract
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        elif isinstance(frame, Image.Image):
            pil_image = frame # Already a PIL image
        else:
            logger.error(f"Unsupported frame type for OCR: {type(frame)}")
            return []


        results = []

        if regions:
            # Process each specified region
            for x, y, w, h in regions:
                # Crop the region from the PIL image
                region_image = pil_image.crop((x, y, x + w, y + h))
                # Get OCR data including confidence
                ocr_data = pytesseract.image_to_data(
                    region_image, lang=lang, output_type=pytesseract.Output.DICT
                )

                # Extract text and confidence (handle cases with no detected text)
                full_text = ""
                confidences = [int(c) for c in ocr_data['conf'] if int(c) > -1] # Filter placeholder -1 confidences
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

                # Reconstruct text from words
                current_block = -1
                block_text = ""
                for i, text_word in enumerate(ocr_data['text']):
                    if text_word.strip() and int(ocr_data['conf'][i]) > 0: # Check confidence > 0
                        if ocr_data['block_num'][i] != current_block:
                            if block_text:
                                full_text += block_text.strip() + "\n" # Add newline between blocks
                            block_text = ""
                            current_block = ocr_data['block_num'][i]
                        block_text += text_word + " "

                if block_text: # Add last block
                    full_text += block_text.strip()

                full_text = full_text.strip()

                if full_text:
                    results.append(
                        {
                            "text": full_text,
                            "bbox": (x, y, w, h),
                            "confidence": avg_confidence, # Use average confidence
                        }
                    )
        else:
            # Process entire frame
            ocr_data = pytesseract.image_to_data(
                pil_image, lang=lang, output_type=pytesseract.Output.DICT
            )
            # Extract text and confidence similarly to regions
            full_text = ""
            confidences = [int(c) for c in ocr_data['conf'] if int(c) > -1]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            current_block = -1
            block_text = ""
            for i, text_word in enumerate(ocr_data['text']):
                 if text_word.strip() and int(ocr_data['conf'][i]) > 0:
                    if ocr_data['block_num'][i] != current_block:
                        if block_text:
                            full_text += block_text.strip() + "\n"
                        block_text = ""
                        current_block = ocr_data['block_num'][i]
                    block_text += text_word + " "

            if block_text:
                full_text += block_text.strip()

            full_text = full_text.strip()

            if full_text:
                results.append(
                    {
                        "text": full_text,
                        "bbox": None, # No specific bbox for full frame
                        "confidence": avg_confidence,
                    }
                )

        return results

    except pytesseract.TesseractNotFoundError:
        logger.critical("Tesseract is not installed or not in your PATH.")
        return []
    except Exception as e:
        logger.error(f"OCR processing error: {str(e)}")
        return []


def detect_software_names(frame, software_list, lang="eng"):
    """
    Detects specific software application names within a video frame using OCR.

    This function orchestrates the OCR process:
    1. Preprocesses the frame for better text visibility.
    2. Detects potential text regions using MSER.
    3. Performs OCR on the detected regions.
    4. Compares the recognized text (case-insensitive) against a provided list
       of target software names.

    Args:
        frame (np.ndarray): The input video frame (BGR format from OpenCV).
        software_list (list[str]): A list of software names to search for.
        lang (str, optional): The language code for Tesseract OCR. Defaults to "eng".

    Returns:
        list[dict]: A list of dictionaries for each detected software match,
                    containing:
                    - 'software' (str): The name of the software from the input list that was matched.
                    - 'detected_text' (str): The actual text segment recognized by OCR that contained the match.
                    - 'confidence' (float): The confidence score from the OCR result for that text segment.
                    - 'location' (tuple[int, int, int, int]): The bounding box (x, y, w, h) of the text region.
    """
    # 1. Preprocess the frame
    processed_frame = preprocess_frame(frame)

    # 2. Detect text regions
    # Consider passing the original frame if MSER works better on non-binary images
    regions = detect_text_regions(processed_frame)
    if not regions:
        logger.debug("No text regions detected in frame.")
        # Optional: Fallback to OCR on the whole frame if no regions found
        # ocr_results = perform_ocr(processed_frame, regions=None, lang=lang)
        return [] # Or proceed with fallback

    # 3. Perform OCR on detected regions using the processed frame
    ocr_results = perform_ocr(processed_frame, regions=regions, lang=lang)
    if not ocr_results:
        logger.debug("OCR performed, but no text recognized in regions.")
        return []

    # 4. Match detected text against software list
    matches = []
    for result in ocr_results:
        text_lower = result["text"].lower()
        for software in software_list:
            software_lower = software.lower()
            # Use 'in' for substring matching (e.g., "photoshop" in "adobe photoshop")
            if software_lower in text_lower:
                logger.info(f"Found potential match: '{software}' in OCR text '{result['text']}'")
                matches.append(
                    {
                        "software": software,
                        "detected_text": result["text"],
                        "confidence": result["confidence"],
                        "location": result["bbox"], # Bbox from the region OCR'd
                    }
                )
                # Optional: break if only one match per region is desired

    return matches
