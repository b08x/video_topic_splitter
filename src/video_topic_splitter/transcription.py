#!/usr/bin/env python3
"""Transcription services for video topic splitter."""

import json
import os
import time
from typing import Any, Dict

from deepgram import (DeepgramClient, DeepgramError, FileSource,
                      PrerecordedOptions)
from groq import Groq


def transcribe_file_deepgram(
    client: DeepgramClient,
    file_path: str,
    options: PrerecordedOptions,
    max_retries: int = 3,
    retry_delay: int = 5,
) -> Dict[str, Any]:
    """Transcribe audio file using Deepgram API.

    Args:
        client (DeepgramClient): Initialized Deepgram client object.
        file_path (str): Path to the audio file.
        options (PrerecordedOptions): Deepgram transcription options.
        max_retries (int, optional): Maximum number of retries on API failure. Defaults to 3.
        retry_delay (int, optional): Delay in seconds between retries. Defaults to 5.

    Returns:
        Dict[str, Any]: The JSON response from the Deepgram API.  Raises exception if transcription fails after retries.

    Raises:
        DeepgramError: If the Deepgram API call fails after multiple retries.
        Exception: For any other unexpected errors during transcription.
    """
    print("Transcribing audio using Deepgram...")
    for attempt in range(max_retries):
        try:
            with open(file_path, "rb") as audio:
                buffer_data = audio.read()
                payload: FileSource = {
                    "buffer": buffer_data,
                    "mimetype": "audio/mp4",
                }  # Assuming mp4, adjust if needed
                response = client.listen.rest.v("1").transcribe_file(payload, options)
            print("Transcription complete.")
            return json.loads(response.to_json())
        except DeepgramError as e:
            if attempt < max_retries - 1:
                print(
                    f"API call failed. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(retry_delay)
            else:
                print(f"Transcription failed after {max_retries} attempts: {str(e)}")
                raise
        except Exception as e:
            print(f"Unexpected error during transcription: {str(e)}")
            raise


def transcribe_file_groq(
    client: Groq,
    file_path: str,
    model: str = "whisper-large-v3",
    language: str = "en",
    prompt: str = None,
) -> Dict[str, Any]:
    """Transcribe audio file using Groq API.

    Args:
        client (Groq): Initialized Groq client object.
        file_path (str): Path to the audio file.
        model (str, optional): The Groq transcription model to use. Defaults to "whisper-large-v3".
        language (str, optional): The language of the audio. Defaults to "en".
        prompt (str, optional): An optional prompt to guide the transcription. Defaults to None.

    Returns:
        Dict[str, Any]: The JSON response from the Groq API.

    Raises:
        Exception: For any errors during Groq transcription.
    """
    print("Transcribing audio using Groq...")
    try:
        with open(file_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(file_path, file.read()),
                model=model,
                prompt=prompt,
                response_format="verbose_json",
                language=language,
                temperature=0.2,  # Adjust temperature as needed
            )
        print("Transcription complete.")
        return json.loads(transcription.text)
    except Exception as e:
        print(f"Error during Groq transcription: {str(e)}")
        raise


def save_transcription(transcription: Dict[str, Any], project_path: str):
    """Save raw transcription (typically from Deepgram or similar API) to a JSON file.

    Args:
        transcription (Dict[str, Any]): The raw transcription data.
        project_path (str): The path to the project directory.
    """
    transcription_path = os.path.join(project_path, "transcription.json")
    with open(transcription_path, "w") as f:
        json.dump(transcription, f, indent=2)
    print(f"Transcription saved to: {transcription_path}")


def save_transcript(transcript: Dict[str, Any], project_path: str):
    """Save processed transcript data to a JSON file.

    Args:
        transcript (Dict[str, Any]): The processed transcript data.
        project_path (str): The path to the project directory.
    """
    transcript_path = os.path.join(project_path, "transcript.json")
    with open(transcript_path, "w") as f:
        json.dump(transcript, f, indent=2)
    print(f"Transcript saved to: {transcript_path}")


def load_transcript(transcript_path: str) -> Dict[str, Any]:
    """Load a processed transcript from a JSON file.

    Args:
        transcript_path (str): Path to the transcript JSON file.

    Returns:
        Dict[str, Any]: The loaded transcript data.

    Raises:
        FileNotFoundError: If the transcript file does not exist.
        json.JSONDecodeError: If the JSON file is invalid.
    """
    with open(transcript_path, "r") as f:
        return json.load(f)
