# api/openai.py
#!/usr/bin/env python3
"""OpenAI API integration for transcription services using cURL."""

import json
import os
import subprocess


def transcribe_with_whisper(audio_path: str) -> dict:
    """
    Transcribes an audio file using the OpenAI Whisper API via a cURL command.

    This function constructs and executes a cURL command to send the audio file
    to the Whisper API for transcription. It supports custom API endpoints through
    the `OPENAI_API_BASE` environment variable and handles optional API keys,
    making it suitable for use with local Whisper servers.

    Args:
        audio_path (str): The path to the audio file to be transcribed.

    Returns:
        dict: The JSON response from the Whisper API as a Python dictionary.

    Raises:
        RuntimeError: If the cURL command fails or returns a non-zero exit code.
        json.JSONDecodeError: If the output from cURL is not valid JSON.
    """
    print("Transcribing audio with Whisper API...")

    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    endpoint = f"{api_base}/audio/transcriptions"

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Construct the cURL command
    curl_command = [
        "curl",
        "-s",  # Silent mode
        "-X",
        "POST",
        endpoint,
        "-F",
        f"file=@{audio_path}",
        "-F",
        "model=whisper-1",
        "-F",
        "response_format=verbose_json",
    ]

    # Add headers
    for key, value in headers.items():
        curl_command.extend(["-H", f"{key}: {value}"])

    try:
        # Execute the command
        process = subprocess.run(
            curl_command,
            check=True,
            capture_output=True,
            text=True,
        )
        response_json = json.loads(process.stdout)
        print("Transcription complete.")
        return response_json

    except subprocess.CalledProcessError as e:
        error_message = (
            f"cURL command failed with exit code {e.returncode}.\n"
            f"Stderr: {e.stderr}\n"
            f"Stdout: {e.stdout}"
        )
        print(error_message)
        raise RuntimeError(error_message)
    except json.JSONDecodeError as e:
        error_message = f"Failed to decode JSON from cURL output: {e}"
        print(error_message)
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise
