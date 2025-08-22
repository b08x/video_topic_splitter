# api/openai.py
#!/usr/bin/env python3
"""
OpenAI API integration for transcription services.

This module provides a function to interact with the OpenAI Whisper API for
audio transcription. It uses the official OpenAI Python library for secure
and reliable API communication, supporting custom API endpoints and optional
API keys for use with local or alternative Whisper server implementations.
"""

import os

import openai


def transcribe_with_whisper(audio_path: str) -> dict:
    """
    Transcribes an audio file using the OpenAI Whisper API via the official OpenAI Python library.

    This function uses the official OpenAI client library to send the audio file
    to the Whisper API for transcription. It supports custom API endpoints through
    the `OPENAI_API_BASE` environment variable and handles optional API keys,
    making it suitable for use with local Whisper servers.

    Args:
        audio_path (str): The path to the audio file to be transcribed.

    Returns:
        dict: The JSON response from the Whisper API as a Python dictionary.

    Raises:
        openai.APIError: If the API request fails due to an API-specific error.
        openai.RateLimitError: If the API request is rate limited.
        openai.AuthenticationError: If the API key is invalid or missing when required.
        FileNotFoundError: If the specified audio file does not exist.
        Exception: For any other unexpected errors during transcription.
    """
    print("Transcribing audio with Whisper API...")

    # Get configuration from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

    # Initialize the OpenAI client with custom configuration
    client = openai.OpenAI(
        api_key=api_key,
        base_url=api_base
    )

    try:
        # Verify the audio file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Open and transcribe the audio file
        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                response_format="verbose_json"
            )

        # Convert the response to a dictionary for consistency with the original function
        response_dict = response.model_dump()
        print("Transcription complete.")
        return response_dict

    except openai.AuthenticationError as e:
        error_message = f"OpenAI API authentication failed. Please check your API key: {e}"
        print(error_message)
        raise openai.AuthenticationError(error_message) from e

    except openai.RateLimitError as e:
        error_message = f"OpenAI API rate limit exceeded. Please try again later: {e}"
        print(error_message)
        raise openai.RateLimitError(error_message) from e

    except openai.APIError as e:
        error_message = f"OpenAI API error occurred: {e}"
        print(error_message)
        raise openai.APIError(error_message) from e

    except FileNotFoundError as e:
        error_message = f"Audio file not found: {e}"
        print(error_message)
        raise

    except Exception as e:
        error_message = f"An unexpected error occurred during transcription: {e}"
        print(error_message)
        raise
