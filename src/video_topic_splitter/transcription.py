#!/usr/bin/env python3
"""
Transcription services for video topic splitter, including loading from files
and generating transcripts via API.
"""

import json
import os
import re
from typing import Dict, List, Optional

from .api.openai import transcribe_with_whisper


def time_to_seconds(time_str: str) -> float:
    """
    Convert a timestamp string (HH:MM:SS,ms or MM:SS,ms) to seconds.

    Args:
        time_str: The timestamp string.

    Returns:
        The total number of seconds as a float.
    """
    parts = time_str.replace(",", ".").split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return 0.0


def parse_srt(content: str) -> List[Dict]:
    """
    Parse the content of an SRT subtitle file into a standard transcript format.

    This function uses a regular expression to robustly parse SRT segments,
    including their index, start and end times, and text content.

    Args:
        content: The string content of the SRT file.

    Returns:
        A list of dictionaries, where each dictionary represents a transcript
        segment with 'start', 'end', and 'content' keys.
    """
    transcript = []
    # Regex to capture segment index, start/end times, and text
    segment_pattern = re.compile(
        r"(\d+)\n"  # Segment index
        r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n"  # Timestamps
        r"((?:.+\n?)+)",  # Text (can be multi-line)
        re.MULTILINE,
    )
    for match in segment_pattern.finditer(content):
        start_time = time_to_seconds(match.group(2))
        end_time = time_to_seconds(match.group(3))
        text = match.group(4).strip().replace("\n", " ")
        transcript.append({"start": start_time, "end": end_time, "content": text})
    return transcript


def parse_vtt(content: str) -> List[Dict]:
    """
    Parse the content of a VTT subtitle file into a standard transcript format.

    This function handles the VTT format, skipping metadata headers and parsing
    each cue to extract start time, end time, and text content.

    Args:
        content: The string content of the VTT file.

    Returns:
        A list of dictionaries, where each dictionary represents a transcript
        segment with 'start', 'end', and 'content' keys.
    """
    transcript = []
    # VTT can have metadata headers, so we skip to the first cue
    cues = content.split("\n\n")
    for cue in cues:
        if "-->" in cue:
            lines = cue.split("\n")
            time_line = lines[0]
            text_lines = lines[1:]
            try:
                # Extract times, ignoring potential cue IDs or settings
                time_match = re.search(
                    r"(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})", time_line
                )
                if not time_match:
                    # Also handle format without hours
                    time_match = re.search(
                        r"(\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}\.\d{3})", time_line
                    )
                if time_match:
                    start_time = time_to_seconds(time_match.group(1).replace(".", ","))
                    end_time = time_to_seconds(time_match.group(2).replace(".", ","))
                    text = " ".join(text_lines).strip()
                    transcript.append(
                        {"start": start_time, "end": end_time, "content": text}
                    )
            except (IndexError, ValueError) as e:
                print(f"Skipping malformed VTT cue: {cue} - Error: {e}")
                continue
    return transcript


def load_transcript(transcript_path: str) -> List[Dict]:
    """
    Load a transcript from a file (.srt, .vtt, or .json).

    Args:
        transcript_path: Path to the transcript file.

    Returns:
        A list of segment dictionaries.
    """
    _, extension = os.path.splitext(transcript_path)
    with open(transcript_path, "r", encoding="utf-8") as f:
        content = f.read()

    if extension == ".srt":
        return parse_srt(content)
    elif extension == ".vtt":
        return parse_vtt(content)
    elif extension == ".json":
        # Assuming the JSON is already in the desired format
        return json.loads(content)
    else:
        raise ValueError(f"Unsupported transcript format: {extension}")


def get_transcript_from_audio(audio_path: str, project_path: str) -> List[Dict]:
    """
    Orchestrates transcription of an audio file using the Whisper API.

    Args:
        audio_path: Path to the audio file.
        project_path: The path to the project directory for saving outputs.

    Returns:
        A list of segment dictionaries in the standard format.
    """
    # Get raw transcription response from Whisper
    transcription_response = transcribe_with_whisper(audio_path)

    # Save the raw response
    save_transcription(transcription_response, project_path)

    # Convert the response to the standard internal transcript format
    transcript = [
        {
            "content": segment["text"].strip(),
            "start": segment["start"],
            "end": segment["end"],
        }
        for segment in transcription_response.get("segments", [])
    ]

    # Save the processed transcript
    save_transcript(transcript, project_path)
    return transcript


def save_transcription(transcription: dict, project_path: str):
    """
    Save the raw transcription response from the API to a JSON file.

    This is useful for debugging and for later reprocessing of the raw
    transcription data without needing to call the API again.

    Args:
        transcription: The raw JSON response from the transcription API.
        project_path: The path to the project directory.
    """
    transcription_path = os.path.join(project_path, "transcription.json")
    with open(transcription_path, "w") as f:
        json.dump(transcription, f, indent=2)
    print(f"Raw transcription saved to: {transcription_path}")


def save_transcript(transcript: List[Dict], project_path: str):
    """
    Save the processed transcript to a JSON file in the standard format.

    Args:
        transcript: A list of segment dictionaries in the standard format.
        project_path: The path to the project directory.
    """
    transcript_path = os.path.join(project_path, "transcript.json")
    with open(transcript_path, "w") as f:
        json.dump(transcript, f, indent=2)
    print(f"Processed transcript saved to: {transcript_path}")


def seconds_to_timestamp(seconds: float, separator: str = ",") -> str:
    """
    Convert a duration in seconds to a standard SRT/VTT timestamp format.

    Args:
        seconds: The duration in seconds.
        separator: The separator to use between seconds and milliseconds
            ("," for SRT, "." for VTT).

    Returns:
        A formatted timestamp string (HH:MM:SS,ms or HH:MM:SS.ms).
    """
    h = int(seconds / 3600)
    m = int((seconds % 3600) / 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{separator}{ms:03d}"


def save_transcript_to_srt(transcript: List[Dict], project_path: str):
    """
    Save a transcript in the SRT subtitle format.

    Args:
        transcript: A list of segment dictionaries in the standard format.
        project_path: The path to the project directory where the SRT file
            will be saved.
    """
    srt_path = os.path.join(project_path, "transcript.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(transcript, 1):
            start_time = seconds_to_timestamp(segment["start"], separator=",")
            end_time = seconds_to_timestamp(segment["end"], separator=",")
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{segment['content']}\n\n")
    print(f"SRT transcript saved to: {srt_path}")


def save_transcript_to_vtt(transcript: List[Dict], project_path: str):
    """
    Save a transcript in the VTT subtitle format.

    Args:
        transcript: A list of segment dictionaries in the standard format.
        project_path: The path to the project directory where the VTT file
            will be saved.
    """
    vtt_path = os.path.join(project_path, "transcript.vtt")
    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for segment in transcript:
            start_time = seconds_to_timestamp(segment["start"], separator=".")
            end_time = seconds_to_timestamp(segment["end"], separator=".")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{segment['content']}\n\n")
    print(f"VTT transcript saved to: {vtt_path}")
