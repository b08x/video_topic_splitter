# api/deepgram.py
#!/usr/bin/env python3

"""Deepgram API integration for transcription services."""

import json
import time
from typing import Any, Dict

from deepgram import (DeepgramClient, DeepgramError, FileSource,
                      PrerecordedOptions)


def transcribe_file_deepgram(
    client: DeepgramClient,
    file_path: str,
    options: PrerecordedOptions,
    max_retries: int = 3,
    retry_delay: int = 5,
) -> Dict[str, Any]:
    """Transcribe an audio file using the Deepgram Prerecorded API with retry logic.

    This function attempts to transcribe the audio file specified by `file_path`
    using the provided Deepgram client and options. It includes a retry mechanism
    to handle transient API errors.

    Args:
        client: An initialized DeepgramClient instance.
        file_path: The path to the audio file to be transcribed.
        options: A PrerecordedOptions object containing transcription parameters
                 (e.g., model, language, features).
        max_retries: The maximum number of times to retry the API call upon failure.
                     Defaults to 3.
        retry_delay: The delay in seconds between retry attempts. Defaults to 5.

    Returns:
        A dictionary containing the transcription results parsed from the
        Deepgram API JSON response.

    Raises:
        DeepgramError: If the API call fails after the maximum number of retries.
        FileNotFoundError: If the specified `file_path` does not exist.
        Exception: For any other unexpected errors during the process.

    Example:
        >>> from deepgram import DeepgramClient, PrerecordedOptions
        >>> client = DeepgramClient("YOUR_DEEPGRAM_API_KEY")
        >>> options = PrerecordedOptions(model="nova-2", smart_format=True)
        >>> try:
        ...     result = transcribe_file_deepgram(client, "audio.mp4", options)
        ...     print(result)
        ... except Exception as e:
        ...     print(f"Error: {e}")
    """
    print("Transcribing audio using Deepgram...")
    for attempt in range(max_retries):
        try:
            with open(file_path, "rb") as audio:
                buffer_data = audio.read()
                # TODO: Determine mimetype dynamically instead of hardcoding 'audio/mp4'
                #       Consider using the 'mimetypes' module or a library like 'python-magic'.
                payload: FileSource = {"buffer": buffer_data, "mimetype": "audio/mp4"}
                response = client.listen.rest.v("1").transcribe_file(payload, options)
            print("Transcription complete.")
            # The response object has a convenient .to_json() method
            return json.loads(response.to_json(indent=4)) # Added indent for readability if needed elsewhere
        except DeepgramError as e:
            if attempt < max_retries - 1:
                print(
                    f"API call failed. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(retry_delay)
            else:
                print(f"Transcription failed after {max_retries} attempts: {str(e)}")
                raise
        except FileNotFoundError:
            print(f"Error: Audio file not found at {file_path}")
            raise
        except Exception as e:
            print(f"Unexpected error during transcription: {str(e)}")
            raise
    # This line should technically be unreachable due to the raise in the loop,
    # but returning an empty dict or None might be considered depending on desired error handling.
    # However, raising the exception is generally better practice.
    raise RuntimeError("Transcription failed unexpectedly after exhausting retries.") # Added for clarity
