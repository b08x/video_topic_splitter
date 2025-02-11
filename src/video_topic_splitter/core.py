#!/usr/bin/env python3
"""Core processing functionality for video topic splitter."""

import json
import os

import videogrep
from deepgram import DeepgramClient, PrerecordedOptions
from dotenv import load_dotenv
from groq import Groq

from .audio import (convert_to_mono_and_resample, extract_audio,
                    normalize_audio, remove_silence)
from .constants import CHECKPOINTS
from .project import save_checkpoint
from .prompt_templates import get_analysis_prompt, get_topic_prompt
from .prosodic_analysis import ProsodyAnalyzer
from .topic_modeling import process_transcript
from .transcription import (save_transcript, save_transcription,
                            transcribe_file_deepgram, transcribe_file_groq)
from .video_analysis import split_and_analyze_video
from .youtube import download_video

load_dotenv()


def handle_audio_video(video_path, project_path, skip_unsilence=False):
    """Processes audio from video, handling normalization, silence removal, extraction, and conversion.

    Args:
        video_path (str): Path to the video file.
        project_path (str): Path to the project directory.
        skip_unsilence (bool, optional): Whether to skip silence removal. Defaults to False.

    Returns:
        tuple: A tuple containing the paths to the unsilenced video and the mono/resampled audio.

    Raises:
        RuntimeError: If audio normalization or conversion fails.
    """
    audio_dir = os.path.join(project_path, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    # Define paths
    normalized_video_path = os.path.join(project_path, "normalized_video.mkv")
    unsilenced_video_path = os.path.join(project_path, "unsilenced_video.mkv")
    raw_audio_path = os.path.join(audio_dir, "extracted_audio.wav")
    mono_resampled_audio_path = os.path.join(audio_dir, "mono_resampled_audio.m4a")

    # Check for existing processed files
    if os.path.exists(unsilenced_video_path) and os.path.exists(
        mono_resampled_audio_path
    ):
        print("Found existing processed audio files. Using cached versions.")
        return unsilenced_video_path, mono_resampled_audio_path

    # Normalize audio if needed
    if not os.path.exists(normalized_video_path):
        print("Normalizing audio...")
        normalize_result = normalize_audio(video_path, normalized_video_path)
        if normalize_result["status"] == "error":
            print(f"Error during audio normalization: {normalize_result['message']}")
            raise RuntimeError("Audio normalization failed")
        else:
            print(normalize_result["message"])
    else:
        print("Using existing normalized video file.")

    # Handle silence removal based on skip_unsilence flag
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
                print(
                    f"Error during silence removal: {silence_removal_result['message']}"
                )
                raise RuntimeError("Silence removal failed")
            else:
                print(silence_removal_result["message"])
    # Extract audio if needed
    if not os.path.exists(raw_audio_path):
        print("Extracting audio from video...")
        try:
            extract_audio(unsilenced_video_path, raw_audio_path)
            print("Audio extraction complete.")
        except Exception as e:
            print(f"Error during audio extraction: {str(e)}")
            raise

    # Convert to mono and resample if needed
    if not os.path.exists(mono_resampled_audio_path):
        print("Converting audio to mono and resampling...")
        conversion_result = convert_to_mono_and_resample(
            raw_audio_path, mono_resampled_audio_path
        )
        if conversion_result["status"] == "error":
            print(f"Error during audio conversion: {conversion_result['message']}")
            raise RuntimeError("Audio conversion failed")
        else:
            print(conversion_result["message"])
    else:
        print("Using existing mono resampled audio file.")

    # Save checkpoint only if we've successfully processed everything
    if os.path.exists(unsilenced_video_path) and os.path.exists(
        mono_resampled_audio_path
    ):
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


def process_prosodic_features(
    audio_path: str, transcript: list, project_path: str
) -> dict:
    """Extracts and aligns prosodic features with the transcript.

    Args:
        audio_path (str): Path to the audio file.
        transcript (list): List of transcript segments.
        project_path (str): Path to the project directory.

    Returns:
        dict: A dictionary containing the extracted prosodic features and the aligned segments.
            Keys: 'features', 'aligned_segments'.

    Raises:
        Exception: If prosodic feature processing fails.
    """
    try:
        prosody_analyzer = ProsodyAnalyzer()

        # Extract prosodic features
        features = prosody_analyzer.extract_prosodic_features(audio_path)

        # Save raw features
        features_path = os.path.join(project_path, "prosodic_features.json")
        with open(features_path, "w") as f:
            json.dump(features, f, indent=2)

        save_checkpoint(
            project_path,
            CHECKPOINTS["PROSODIC_FEATURES_EXTRACTED"],
            {"features_path": features_path},
        )

        # Align features with transcript
        aligned_segments = prosody_analyzer.align_features_with_transcript(
            features, transcript
        )

        # Save aligned features
        aligned_path = os.path.join(project_path, "aligned_features.json")
        with open(aligned_path, "w") as f:
            json.dump(aligned_segments, f, indent=2)

        save_checkpoint(
            project_path,
            CHECKPOINTS["FEATURES_ALIGNED"],
            {"aligned_path": aligned_path},
        )

        return {"features": features, "aligned_segments": aligned_segments}

    except Exception as e:
        print(f"Error processing prosodic features: {str(e)}")
        raise


def handle_transcription(
    video_path,
    audio_path,
    project_path,
    api="deepgram",
    num_topics=2,
    groq_prompt=None,
    software_list=None,
    logo_db_path=None,
    ocr_lang="eng",
    logo_threshold=0.8,
    thumbnail_interval=5,
    max_thumbnails=5,
    min_thumbnail_confidence=0.7,
    register="gen-ai",
):
    """Handles transcription of video/audio content using Deepgram or Groq.

    Args:
        video_path (str): Path to the video file.
        audio_path (str): Path to the audio file.
        project_path (str): Path to the project directory.
        api (str, optional): Transcription API to use ("deepgram" or "groq"). Defaults to "deepgram".
        num_topics (int, optional): Number of topics for topic modeling. Defaults to 2.
        groq_prompt (str, optional): Custom prompt for Groq API. Defaults to None.
        software_list (list, optional): List of software to detect. Defaults to None.
        logo_db_path (str, optional): Path to the logo database. Defaults to None.
        ocr_lang (str, optional): Language for OCR. Defaults to "eng".
        logo_threshold (float, optional): Threshold for logo detection. Defaults to 0.8.
        thumbnail_interval (int, optional): Interval for thumbnail extraction (seconds). Defaults to 5.
        max_thumbnails (int, optional): Maximum number of thumbnails per segment. Defaults to 5.
        min_thumbnail_confidence (float, optional): Minimum confidence for thumbnail object detection. Defaults to 0.7.
        register (str, optional): Language register for prompts. Defaults to "gen-ai".

    Returns:
        dict: Results of the transcription and analysis.

    Raises:
        ValueError: If API key is not set.
    """
    segments_dir = os.path.join(project_path, "segments")
    os.makedirs(segments_dir, exist_ok=True)

    print("Parsing transcript with Videogrep...")
    transcript = videogrep.parse_transcript(video_path)
    print("Transcript parsing complete.")

    if not transcript:
        print("No transcript found. Transcribing audio...")
        deepgram_key = os.getenv("DG_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")

        if not deepgram_key:
            raise ValueError("DG_API_KEY environment variable is not set")
        if not groq_key and api == "groq":
            raise ValueError("GROQ_API_KEY environment variable is not set")

        if api == "deepgram":
            deepgram_client = DeepgramClient(deepgram_key)
            deepgram_options = PrerecordedOptions(
                model="nova-2",
                language="en",
                topics=True,
                intents=True,
                smart_format=True,
                punctuate=True,
                paragraphs=True,
                utterances=True,
                diarize=True,
                filler_words=True,
                sentiment=True,
            )
            transcription = transcribe_file_deepgram(
                deepgram_client, audio_path, deepgram_options
            )
            transcript = [
                {
                    "content": utterance["transcript"],
                    "start": utterance["start"],
                    "end": utterance["end"],
                }
                for utterance in transcription["results"]["utterances"]
            ]
        else:  # Groq
            groq_client = Groq(api_key=groq_key)
            # Use register-specific prompt if no custom prompt provided
            if not groq_prompt:
                groq_prompt = get_topic_prompt(
                    register,
                    "Transcribe with focus on technical details and terminology.",
                )
            transcription = transcribe_file_groq(
                groq_client, audio_path, prompt=groq_prompt
            )
            transcript = [
                {
                    "content": segment["text"],
                    "start": segment["start"],
                    "end": segment["end"],
                }
                for segment in transcription["segments"]
            ]

        save_transcription(transcription, project_path)
        save_transcript(transcript, project_path)

    save_checkpoint(
        project_path, CHECKPOINTS["TRANSCRIPTION_COMPLETE"], {"transcript": transcript}
    )

    # Process prosodic features before topic modeling
    try:
        prosodic_results = process_prosodic_features(
            audio_path, transcript, project_path
        )
        # Add prosodic features to transcript segments
        for i, segment in enumerate(transcript):
            if i < len(prosodic_results["aligned_segments"]):
                segment["prosodic_features"] = prosodic_results["aligned_segments"][i][
                    "prosodic_features"
                ]
    except Exception as e:
        print(f"Warning: Prosodic feature extraction failed: {str(e)}")
        prosodic_results = None

    # Process transcript with prosodic features
    results = process_transcript(
        transcript, project_path, num_topics, register=register
    )
    if prosodic_results:
        results["prosodic_features"] = prosodic_results["features"]

    # Split the video and analyze segments
    try:
        analyzed_segments = split_and_analyze_video(
            video_path,
            results["segments"],
            segments_dir,
            software_list,
            logo_db_path,
            ocr_lang,
            logo_threshold,
            thumbnail_interval,
            max_thumbnails,
            min_thumbnail_confidence,
            register=register,
        )

        # Update results with analyzed segments
        results["analyzed_segments"] = analyzed_segments

        # Save results after successful analysis
        results_path = os.path.join(project_path, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Successfully analyzed {len(analyzed_segments)} segments")
    except Exception as e:
        print(f"Error during video analysis: {str(e)}")
        # Load any segments that were successfully analyzed
        try:
            analyzed_segments = split_and_analyze_video(
                video_path, [], segments_dir
            )  # Just load existing
            results["analyzed_segments"] = analyzed_segments
            print(f"Recovered {len(analyzed_segments)} previously analyzed segments")
        except Exception as load_error:
            print(f"Could not load analyzed segments: {str(load_error)}")
            results["analyzed_segments"] = []

    # Save updated results
    results_path = os.path.join(project_path, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    save_checkpoint(project_path, CHECKPOINTS["VIDEO_ANALYZED"], {"results": results})

    return results


def process_video(
    video_path,
    project_path,
    api="deepgram",
    num_topics=2,
    groq_prompt=None,
    skip_unsilence=False,
    transcribe_only=False,
    is_youtube_url=False,
    software_list=None,
    logo_db_path=None,
    ocr_lang="eng",
    logo_threshold=0.8,
    thumbnail_interval=5,
    max_thumbnails=10,
    min_thumbnail_confidence=0.7,
    register="gen-ai",
):
    """Main video processing pipeline, orchestrating audio/video processing, transcription, and analysis.

    Args:
        video_path (str): Path to the video file or YouTube URL.
        project_path (str): Path to the project directory.
        api (str, optional): Transcription API to use ("deepgram" or "groq"). Defaults to "deepgram".
        num_topics (int, optional): Number of topics for topic modeling. Defaults to 2.
        groq_prompt (str, optional): Custom prompt for Groq API. Defaults to None.
        skip_unsilence (bool, optional): Whether to skip silence removal. Defaults to False.
        transcribe_only (bool, optional): Whether to only transcribe the video. Defaults to False.
        is_youtube_url (bool, optional): Whether the video_path is a YouTube URL. Defaults to False.
        software_list (list, optional): List of software to detect. Defaults to None.
        logo_db_path (str, optional): Path to the logo database. Defaults to None.
        ocr_lang (str, optional): Language for OCR. Defaults to "eng".
        logo_threshold (float, optional): Threshold for logo detection. Defaults to 0.8.
        thumbnail_interval (int, optional): Interval for thumbnail extraction (seconds). Defaults to 5.
        max_thumbnails (int, optional): Maximum number of thumbnails per segment. Defaults to 10.
        min_thumbnail_confidence (float, optional): Minimum confidence for thumbnail object detection. Defaults to 0.7.
        register (str, optional): Language register for prompts. Defaults to "gen-ai".


    Returns:
        dict: Results of the processing pipeline, including transcription, topic modeling, and analysis (if not transcribe_only).

    Raises:
        ValueError: If API key is not set.
        RuntimeError: If YouTube download fails.

    """
    from .project import load_checkpoint

    checkpoint = load_checkpoint(project_path)

    # Handle YouTube video download if needed
    if is_youtube_url:
        if (
            checkpoint is None
            or checkpoint["stage"] < CHECKPOINTS["YOUTUBE_DOWNLOAD_COMPLETE"]
        ):
            print("Downloading YouTube video...")
            download_path = os.path.join(project_path, "source_video.mp4")
            result = download_video(
                video_path, download_path, project_path
            )  # Pass project_path for thumbnail download

            if result["status"] == "error":
                raise RuntimeError(f"YouTube download failed: {result['message']}")

            video_path = download_path
            save_checkpoint(
                project_path,
                CHECKPOINTS["YOUTUBE_DOWNLOAD_COMPLETE"],
                {
                    "video_path": video_path,
                    "thumbnail_info": result.get("thumbnail_info"),
                },
            )
            print(result["message"])
        else:
            video_path = checkpoint["data"]["video_path"]
            print("Using previously downloaded YouTube video.")

    if checkpoint is None or checkpoint["stage"] < CHECKPOINTS["AUDIO_PROCESSED"]:
        unsilenced_video_path, mono_resampled_audio_path = handle_audio_video(
            video_path, project_path, skip_unsilence
        )
    else:
        unsilenced_video_path = checkpoint["data"]["unsilenced_video_path"]
        mono_resampled_audio_path = checkpoint["data"]["mono_resampled_audio_path"]

    if checkpoint is None or checkpoint["stage"] < CHECKPOINTS["VIDEO_ANALYZED"]:
        if transcribe_only:
            print("Transcribe-only mode: Skipping topic modeling and video analysis...")
            # Get transcript only
            print("Parsing transcript with Videogrep...")
            transcript = videogrep.parse_transcript(unsilenced_video_path)
            print("Transcript parsing complete.")

            if not transcript:
                print("No transcript found. Transcribing audio...")
                deepgram_key = os.getenv("DG_API_KEY")
                groq_key = os.getenv("GROQ_API_KEY")

                if not deepgram_key:
                    raise ValueError("DG_API_KEY environment variable is not set")
                if not groq_key and api == "groq":
                    raise ValueError("GROQ_API_KEY environment variable is not set")

                if api == "deepgram":
                    deepgram_client = DeepgramClient(deepgram_key)
                    deepgram_options = PrerecordedOptions(
                        model="nova-2",
                        language="en",
                        smart_format=True,
                        punctuate=True,
                        paragraphs=True,
                        utterances=True,
                    )
                    transcription = transcribe_file_deepgram(
                        deepgram_client, mono_resampled_audio_path, deepgram_options
                    )
                    transcript = [
                        {
                            "content": utterance["transcript"],
                            "start": utterance["start"],
                            "end": utterance["end"],
                        }
                        for utterance in transcription["results"]["utterances"]
                    ]
                else:  # Groq
                    groq_client = Groq(api_key=groq_key)
                    # Use register-specific prompt if no custom prompt provided
                    if not groq_prompt:
                        groq_prompt = get_topic_prompt(
                            register,
                            "Transcribe with focus on technical details and terminology.",
                        )
                    transcription = transcribe_file_groq(
                        groq_client, mono_resampled_audio_path, prompt=groq_prompt
                    )
                    transcript = [
                        {
                            "content": segment["text"],
                            "start": segment["start"],
                            "end": segment["end"],
                        }
                        for segment in transcription["segments"]
                    ]

                save_transcription(transcription, project_path)
                save_transcript(transcript, project_path)

            results = {"transcript": transcript, "transcription_only": True}
            save_checkpoint(
                project_path,
                CHECKPOINTS["TRANSCRIBE_ONLY_COMPLETE"],
                {"results": results},
            )
        else:
            results = handle_transcription(
                unsilenced_video_path,
                mono_resampled_audio_path,
                project_path,
                api,
                num_topics,
                groq_prompt,
                software_list,
                logo_db_path,
                ocr_lang,
                logo_threshold,
                thumbnail_interval,
                max_thumbnails,
                min_thumbnail_confidence,
                register,
            )
    else:
        results = checkpoint["data"]["results"]

    final_checkpoint = (
        CHECKPOINTS["TRANSCRIBE_ONLY_COMPLETE"]
        if transcribe_only
        else CHECKPOINTS["PROCESS_COMPLETE"]
    )
    save_checkpoint(project_path, final_checkpoint, {"results": results})

    return results
