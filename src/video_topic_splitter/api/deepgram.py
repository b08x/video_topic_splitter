# api/deepgram.py
#!/usr/bin/env python3

"""Deepgram API integration for transcription services."""

import json
import time

from deepgram import (DeepgramClient, DeepgramError, FileSource,
                      PrerecordedOptions)


def transcribe_file_deepgram(client, file_path, options, max_retries=3, retry_delay=5):
    """Transcribe audio file using Deepgram API."""
    print("Transcribing audio using Deepgram...")
    for attempt in range(max_retries):
        try:
            with open(file_path, "rb") as audio:
                buffer_data = audio.read()
                payload: FileSource = {"buffer": buffer_data, "mimetype": "audio/mp4"}
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
