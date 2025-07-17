#!/usr/bin/env python3
"""Core processing functionality for video topic splitter."""

import json
import os

from dotenv import load_dotenv

from .analysis.topic_modeling import process_transcript
from .analysis.visual_analysis import split_and_analyze_video
from .constants import CHECKPOINTS
from .processing.audio.audio import (convert_to_mono_and_resample,
                                     extract_audio, normalize_audio,
                                     remove_silence)
from .project import save_checkpoint
from .transcription import (get_transcript_from_audio, load_transcript,
                            save_transcript_to_srt, save_transcript_to_vtt)
from .utils.youtube import download_video

load_dotenv()


def handle_audio_video(video_path, project_path, skip_unsilence=False):
    """Process audio from video file with checkpointing."""
    # ... (This function remains largely the same)
    audio_dir = os.path.join(project_path, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    video_name, video_ext = os.path.splitext(os.path.basename(video_path))

    normalized_video_path = os.path.join(
        project_path, f"normalized_video{video_ext}")
    unsilenced_video_path = os.path.join(
        project_path, f"unsilenced_video{video_ext}")
    raw_audio_path = os.path.join(audio_dir, "extracted_audio.opus")
    mono_resampled_audio_path = os.path.join(
        audio_dir, "mono_resampled_audio.m4a")

    if os.path.exists(mono_resampled_audio_path):
        print("Found existing processed audio file. Using cached version.")
        return unsilenced_video_path, mono_resampled_audio_path

    if not os.path.exists(normalized_video_path):
        print("Normalizing audio...")
        normalize_result = normalize_audio(video_path, normalized_video_path)
        if normalize_result["status"] == "error":
            raise RuntimeError(f"Audio normalization failed: {normalize_result['message']}")
        else:
            print(normalize_result["message"])
    else:
        print("Using existing normalized video file.")

    if not os.path.exists(unsilenced_video_path):
        if skip_unsilence:
            import shutil
            shutil.copy2(normalized_video_path, unsilenced_video_path)
            print("Skipping silence removal as requested.")
        else:
            silence_removal_result = remove_silence(
                normalized_video_path, unsilenced_video_path
            )
            if silence_removal_result["status"] == "error":
                raise RuntimeError(f"Silence removal failed: {silence_removal_result['message']}")
            else:
                print(silence_removal_result["message"])

    if not os.path.exists(raw_audio_path):
        print("Extracting audio from video...")
        extract_audio(unsilenced_video_path, raw_audio_path)
        print("Audio extraction complete.")

    if not os.path.exists(mono_resampled_audio_path):
        print("Converting audio to mono and resampling...")
        conversion_result = convert_to_mono_and_resample(
            raw_audio_path, mono_resampled_audio_path
        )
        if conversion_result["status"] == "error":
            raise RuntimeError(f"Audio conversion failed: {conversion_result['message']}")
        else:
            print(conversion_result["message"])
    else:
        print("Using existing mono resampled audio file.")

    save_checkpoint(
        project_path,
        CHECKPOINTS["AUDIO_PROCESSED"],
        {
            "unsilenced_video_path": unsilenced_video_path,
            "mono_resampled_audio_path": mono_resampled_audio_path,
        },
    )
    print("Audio processing checkpoint saved.")
    return unsilenced_video_path, mono_resampled_audio_path


def get_or_create_transcript(
    transcript_path, audio_path, project_path, transcribe_only
):
    """Load or generate a transcript, then save it in multiple formats."""
    if transcript_path:
        print(f"Loading transcript from: {transcript_path}")
        transcript = load_transcript(transcript_path)
    else:
        print("No transcript file provided, starting transcription process...")
        transcript = get_transcript_from_audio(audio_path, project_path)

    # Save transcript in multiple formats regardless of source
    save_transcript_to_srt(transcript, project_path)
    save_transcript_to_vtt(transcript, project_path)

    save_checkpoint(
        project_path,
        CHECKPOINTS["TRANSCRIPTION_COMPLETE"],
        {"transcript": transcript},
    )

    if transcribe_only:
        return {"transcript": transcript, "transcription_only": True}

    return transcript


def process_video(
    video_path,
    project_path,
    transcript_path=None,
    num_topics=5,
    skip_unsilence=False,
    transcribe_only=False,
    is_youtube_url=False,
    software_list=None,
    ocr_lang="eng",
    frames_per_scene=1,
    register="it-workflow",
):
    """Main video processing pipeline."""
    from .project import load_checkpoint

    checkpoint = load_checkpoint(project_path)
    unsilenced_video_path = video_path

    if is_youtube_url:
        # YouTube download logic remains the same...
        if (
            checkpoint is None
            or checkpoint["stage"] < CHECKPOINTS["YOUTUBE_DOWNLOAD_COMPLETE"]
        ):
            print("Downloading YouTube video...")
            download_path = os.path.join(project_path, "source_video.mp4")
            result = download_video(video_path, download_path, project_path)
            if result["status"] == "error":
                raise RuntimeError(f"YouTube download failed: {result['message']}")
            video_path = download_path
            save_checkpoint(
                project_path,
                CHECKPOINTS["YOUTUBE_DOWNLOAD_COMPLETE"],
                {"video_path": video_path, "thumbnail_info": result.get("thumbnail_info")},
            )
            print(result["message"])
        else:
            video_path = checkpoint["data"]["video_path"]
            print("Using previously downloaded YouTube video.")


    mono_resampled_audio_path = None
    if not transcript_path:
        if checkpoint is None or checkpoint["stage"] < CHECKPOINTS["AUDIO_PROCESSED"]:
            unsilenced_video_path, mono_resampled_audio_path = handle_audio_video(
                video_path, project_path, skip_unsilence
            )
        else:
            unsilenced_video_path = checkpoint["data"].get("unsilenced_video_path", video_path)
            mono_resampled_audio_path = checkpoint["data"]["mono_resampled_audio_path"]

    if checkpoint is None or checkpoint["stage"] < CHECKPOINTS["PROCESS_COMPLETE"]:
        transcript = get_or_create_transcript(
            transcript_path, mono_resampled_audio_path, project_path, transcribe_only
        )
        if transcribe_only:
            return transcript

        # Topic modeling remains the same
        topic_results = process_transcript(
            transcript, project_path, num_topics, register=register
        )

        # Visual analysis is now scene-based
        analyzed_scenes = split_and_analyze_video(
            unsilenced_video_path,
            project_path,
            software_list,
            ocr_lang,
            frames_per_scene,
            register,
        )
        
        # Combine results
        results = {
            "topics": topic_results.get("topics", []),
            "segments": topic_results.get("segments", []),
            "analyzed_scenes": analyzed_scenes,
        }

        results_path = os.path.join(project_path, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        save_checkpoint(project_path, CHECKPOINTS["PROCESS_COMPLETE"], {"results": results})
    else:
        results = checkpoint["data"]["results"]

    return results
