#!/usr/bin/env python3

"""
Provides functions for transcribing audio files using external APIs (Deepgram, Groq)
and for saving/loading transcription and transcript data.

This module handles the interaction with transcription services, including error handling
and retries for API calls. It also provides utility functions for persisting
the results to the filesystem.
"""

import os
import json
import time
from typing import Dict, Any, Optional
from deepgram import DeepgramClient, PrerecordedOptions, FileSource, DeepgramError
from groq import Groq

# Assuming CHECKPOINTS is defined elsewhere, potentially for future use or logging.
# from .constants import CHECKPOINTS # Currently unused in this snippet

def transcribe_file_deepgram(
    client: DeepgramClient,
    file_path: str,
    options: PrerecordedOptions,
    max_retries: int = 3,
    retry_delay: int = 5
) -> Dict[str, Any]:
    """
    Transcribes a given audio file using the Deepgram Prerecorded API.

    Attempts to transcribe the audio file specified by file_path using the provided
    Deepgram client and options. Implements a retry mechanism in case of API errors.

    Args:
        client: An initialized DeepgramClient instance.
        file_path: The path to the audio file to be transcribed (e.g., MP4, MP3, WAV).
        options: A PrerecordedOptions object specifying transcription parameters
                 (e.g., model, language, features like diarization, punctuation).
        max_retries: The maximum number of times to retry the API call upon failure.
        retry_delay: The delay in seconds between retry attempts.

    Returns:
        A dictionary containing the transcription results parsed from the
        Deepgram API JSON response.

    Raises:
        DeepgramError: If the transcription fails after all retry attempts.
        FileNotFoundError: If the specified file_path does not exist.
        Exception: For any other unexpected errors during the process.
    """
    print("Transcribing audio using Deepgram...")
    for attempt in range(max_retries):
        try:
            with open(file_path, "rb") as audio:
                buffer_data = audio.read()
                # TODO: Dynamically determine mimetype or make it an argument
                payload: FileSource = {"buffer": buffer_data, "mimetype": "audio/mp4"}
                # Ensure the API version is correct or configurable if needed
                response = client.listen.rest.v("1").transcribe_file(payload, options)
            print("Transcription complete.")
            # The response object might have methods to directly return a dict
            # Consider using response.to_dict() if available and suitable
            return json.loads(response.to_json())
        except DeepgramError as e:
            if attempt < max_retries - 1:
                print(f"API call failed. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                print(f"Transcription failed after {max_retries} attempts: {str(e)}")
                raise
        except FileNotFoundError:
            print(f"Error: Audio file not found at {file_path}")
            raise
        except Exception as e:
            print(f"Unexpected error during Deepgram transcription: {str(e)}")
            raise
    # This line should technically be unreachable due to the raise in the loop,
    # but added for completeness in case of logic changes.
    raise RuntimeError("Transcription failed after exhausting retries, but no exception was propagated.")


def transcribe_file_groq(
    client: Groq,
    file_path: str,
    model: str = "whisper-large-v3",
    language: str = "en",
    prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Transcribes a given audio file using the Groq API (Whisper model).

    Args:
        client: An initialized Groq client instance.
        file_path: The path to the audio file to be transcribed.
        model: The identifier of the Whisper model to use (e.g., "whisper-large-v3").
        language: The language code for the audio (e.g., "en", "es").
        prompt: An optional text prompt to guide the transcription model.

    Returns:
        A dictionary containing the transcription results, typically including
        the transcript text and potentially segment or word timings if requested
        via response_format.

    Raises:
        FileNotFoundError: If the specified file_path does not exist.
        Exception: If any error occurs during the API call or processing.
    """
    print("Transcribing audio using Groq...")
    try:
        with open(file_path, "rb") as file:
            # The file tuple format is specific to how the Groq client expects files.
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(file_path), file.read()), # Pass filename for potential server-side use
                model=model,
                prompt=prompt,
                response_format="verbose_json", # Request detailed output
                language=language,
                temperature=0.2 # Lower temperature for more deterministic output
            )
        print("Transcription complete.")
        # Assuming transcription.text contains the JSON string for verbose_json
        # If the client library parses it directly, adjust accordingly.
        # E.g., if transcription is already a dict-like object: return transcription.to_dict() or similar
        if hasattr(transcription, 'text') and isinstance(transcription.text, str):
             return json.loads(transcription.text)
        elif isinstance(transcription, dict): # Or check if it behaves like a dict
             return transcription
        else:
             # Handle unexpected response format
             print(f"Warning: Unexpected Groq transcription response format: {type(transcription)}")
             # Attempt to return raw if possible, or raise an error
             return {"raw_response": str(transcription)}

    except FileNotFoundError:
        print(f"Error: Audio file not found at {file_path}")
        raise
    except Exception as e:
        print(f"Error during Groq transcription: {str(e)}")
        raise

def save_transcription(transcription: Dict[str, Any], project_path: str) -> None:
    """
    Saves the raw transcription data (typically from the API) to a JSON file.

    Args:
        transcription: The dictionary containing the raw transcription data.
        project_path: The directory path where the 'transcription.json' file
                      will be saved.

    Returns:
        None
    """
    if not os.path.isdir(project_path):
        print(f"Warning: Project directory '{project_path}' does not exist. Attempting to create.")
        try:
            os.makedirs(project_path, exist_ok=True)
        except OSError as e:
            print(f"Error: Could not create directory {project_path}: {e}")
            raise # Or handle appropriately

    transcription_path = os.path.join(project_path, "transcription.json")
    try:
        with open(transcription_path, "w", encoding="utf-8") as f:
            json.dump(transcription, f, indent=2, ensure_ascii=False)
        print(f"Raw transcription saved to: {transcription_path}")
    except IOError as e:
        print(f"Error saving transcription file to {transcription_path}: {e}")
        raise
    except TypeError as e:
        print(f"Error: Data provided to save_transcription is not JSON serializable: {e}")
        raise


def save_transcript(transcript: Any, project_path: str) -> None:
    """
    Saves the processed transcript data to a JSON file.

    This function is intended for saving a potentially structured or cleaned-up
    version of the transcription, which might differ from the raw API output.

    Args:
        transcript: The processed transcript data (e.g., a list of segments,
                    a dictionary, or just the plain text string, though JSON
                    is expected).
        project_path: The directory path where the 'transcript.json' file
                      will be saved.

    Returns:
        None
    """
    if not os.path.isdir(project_path):
        print(f"Warning: Project directory '{project_path}' does not exist. Attempting to create.")
        try:
            os.makedirs(project_path, exist_ok=True)
        except OSError as e:
            print(f"Error: Could not create directory {project_path}: {e}")
            raise # Or handle appropriately

    transcript_path = os.path.join(project_path, "transcript.json")
    try:
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)
        print(f"Processed transcript saved to: {transcript_path}")
    except IOError as e:
        print(f"Error saving transcript file to {transcript_path}: {e}")
        raise
    except TypeError as e:
        print(f"Error: Data provided to save_transcript is not JSON serializable: {e}")
        raise


def load_transcript(transcript_path: str) -> Any:
    """
    Loads transcript data from a specified JSON file.

    Args:
        transcript_path: The full path to the 'transcript.json' file.

    Returns:
        The Python object (typically dict or list) parsed from the JSON file.

    Raises:
        FileNotFoundError: If the transcript_path does not exist.
        json.JSONDecodeError: If the file content is not valid JSON.
        IOError: If there is an error reading the file.
    """
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Transcript file not found at {transcript_path}")
        raise
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from {transcript_path}: {e}")
        raise
    except IOError as e:
        print(f"Error reading transcript file {transcript_path}: {e}")
        raise

