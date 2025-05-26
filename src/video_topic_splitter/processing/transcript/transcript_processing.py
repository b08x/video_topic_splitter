# processing/transcript/transcript_processing.py
#!/usr/bin/env python3
"""Transcript processing utilities.

This module provides functions for loading, processing, and analyzing transcript data
from various sources, including JSON files and API responses.
"""

import json
import logging
from typing import Any, Dict, List, Optional

# Set up logging
logger = logging.getLogger(__name__)


def load_transcript_sentences(path: str) -> List[Dict[str, Any]]:
    """Load transcript sentences from a JSON file.

    Args:
        path (str): Path to the JSON file containing transcript sentences.

    Returns:
        List[Dict[str, Any]]: List of transcript sentence dictionaries.
        Each dictionary typically contains:
        - text: The sentence text
        - start_time: Start time in seconds
        - end_time: End time in seconds
        - speaker: Optional speaker identification

    Raises:
        FileNotFoundError: If the transcript file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    logger.info(f"Loading transcript sentences from: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)

        # Validate the loaded data
        if not isinstance(transcript_data, list):
            logger.warning(
                f"Expected list of sentences, got {type(transcript_data).__name__}. "
                "Attempting to extract sentences."
            )
            # Try to extract sentences from a nested structure
            if isinstance(transcript_data, dict) and "sentences" in transcript_data:
                transcript_data = transcript_data["sentences"]
            else:
                logger.error("Could not extract sentences from transcript data")
                return []

        # Validate each sentence has required fields
        valid_sentences = []
        for i, sentence in enumerate(transcript_data):
            if not isinstance(sentence, dict):
                logger.warning(f"Skipping sentence {i}: not a dictionary")
                continue

            # Check for required fields
            if "text" not in sentence:
                logger.warning(f"Skipping sentence {i}: missing 'text' field")
                continue

            # Ensure start_time and end_time exist (default to 0 if missing)
            if "start_time" not in sentence:
                logger.warning(f"Sentence {i} missing 'start_time', defaulting to 0")
                sentence["start_time"] = 0

            if "end_time" not in sentence:
                logger.warning(f"Sentence {i} missing 'end_time', defaulting to 0")
                sentence["end_time"] = 0

            valid_sentences.append(sentence)

        logger.info(f"Successfully loaded {len(valid_sentences)} transcript sentences")
        return valid_sentences

    except FileNotFoundError:
        logger.error(f"Transcript file not found: {path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in transcript file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading transcript sentences: {e}")
        raise


def merge_sentences_into_paragraphs(
    sentences: List[Dict[str, Any]],
    max_gap: float = 1.0,
    max_paragraph_duration: float = 30.0,
) -> List[Dict[str, Any]]:
    """Merge individual sentences into paragraph-like segments.

    Args:
        sentences (List[Dict[str, Any]]): List of sentence dictionaries
        max_gap (float, optional): Maximum time gap between sentences to be merged.
            Defaults to 1.0 seconds.
        max_paragraph_duration (float, optional): Maximum duration for a paragraph.
            Defaults to 30.0 seconds.

    Returns:
        List[Dict[str, Any]]: List of paragraph dictionaries with merged text
    """
    if not sentences:
        return []

    paragraphs = []
    current_paragraph = {
        "text": sentences[0]["text"],
        "start_time": sentences[0]["start_time"],
        "end_time": sentences[0]["end_time"],
        "sentences": [sentences[0]],
    }

    for sentence in sentences[1:]:
        # Check if this sentence should start a new paragraph
        time_gap = sentence["start_time"] - current_paragraph["end_time"]
        paragraph_duration = (
            current_paragraph["end_time"] - current_paragraph["start_time"]
        )

        # Determine if there's a speaker change
        speaker_change = False
        current_speaker = current_paragraph["sentences"][-1].get("speaker")
        next_speaker = sentence.get("speaker")
        if current_speaker is not None and next_speaker is not None:
            speaker_change = current_speaker != next_speaker

        if (
            time_gap > max_gap
            or paragraph_duration > max_paragraph_duration
            or speaker_change
        ):
            # Finish current paragraph and start a new one
            paragraphs.append(current_paragraph)
            current_paragraph = {
                "text": sentence["text"],
                "start_time": sentence["start_time"],
                "end_time": sentence["end_time"],
                "sentences": [sentence],
            }
        else:
            # Add to current paragraph
            current_paragraph["text"] += " " + sentence["text"]
            current_paragraph["end_time"] = sentence["end_time"]
            current_paragraph["sentences"].append(sentence)

    # Add the last paragraph
    if current_paragraph:
        paragraphs.append(current_paragraph)

    logger.info(f"Merged {len(sentences)} sentences into {len(paragraphs)} paragraphs")
    return paragraphs


def filter_transcript_by_time_range(
    transcript: List[Dict[str, Any]], start_time: float, end_time: float
) -> List[Dict[str, Any]]:
    """Filter transcript sentences to only include those within a time range.

    Args:
        transcript (List[Dict[str, Any]]): List of transcript sentences or paragraphs
        start_time (float): Start time in seconds
        end_time (float): End time in seconds

    Returns:
        List[Dict[str, Any]]: Filtered list of transcript items
    """
    filtered = []

    for item in transcript:
        item_start = item.get("start_time", 0)
        item_end = item.get("end_time", 0)

        # Include if there's any overlap with the target range
        if item_start <= end_time and item_end >= start_time:
            filtered.append(item)

    return filtered
