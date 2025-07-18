#!/usr/bin/env python3
"""
Core processing functionality for the video topic splitter.

This module contains the main pipeline for video analysis, including audio
processing, transcription, topic modeling, and multimodal analysis. It
orchestrates the workflow from input video to final structured output.
"""

import json
import os

from dotenv import load_dotenv

from .analysis.topic_modeling import process_transcript
from .analysis.visual_analysis import split_and_analyze_video
from .analysis.segment_analysis import SegmentProcessor
from .constants import CHECKPOINTS
from .progress_tracker import ProgressTracker, create_console_progress_callback
from .project_structure import ProjectStructure
from .processing.video.video_segmentation import segment_video_by_topics
from .processing.audio.audio import (convert_to_mono_and_resample,
                                     extract_audio, normalize_audio,
                                     remove_silence)
from .project import save_checkpoint
from .transcription import (get_transcript_from_audio, load_transcript,
                            save_transcript_to_srt, save_transcript_to_vtt)
from .utils.youtube import download_video

load_dotenv()


def handle_audio_video(video_path: str, project_path: str, skip_unsilence: bool = False) -> tuple[str, str]:
    """
    Process audio from a video file, including normalization and silence removal.

    This function checks for cached processed audio files to avoid redundant
    processing. It handles audio normalization, silence removal (optional),
    audio extraction, and resampling to a standardized format.

    Args:
        video_path: Path to the input video file.
        project_path: The root directory of the current project.
        skip_unsilence: If True, skips the silence removal step.

    Returns:
        A tuple containing the path to the (potentially unsilenced) video
        and the path to the final mono, resampled audio file.

    Raises:
        RuntimeError: If any of the underlying FFmpeg operations fail.
    """

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
            raise RuntimeError(
                f"Audio normalization failed: {normalize_result['message']}")
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
                raise RuntimeError(
                    f"Silence removal failed: {silence_removal_result['message']}")
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
            raise RuntimeError(
                f"Audio conversion failed: {conversion_result['message']}")
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
    transcript_path: str, audio_path: str, project_path: str, transcribe_only: bool
) -> dict:
    """
    Load a pre-existing transcript or generate one using the speech-to-text service.

    If a transcript path is provided, it loads the file. Otherwise, it
    transcribes the given audio file. The resulting transcript is saved in
    multiple formats (.srt, .vtt) and a checkpoint is created.

    Args:
        transcript_path: Path to a pre-existing transcript file.
        audio_path: Path to the audio file to be transcribed.
        project_path: The root directory of the current project.
        transcribe_only: If True, returns immediately after transcription.

    Returns:
        The transcript data as a dictionary. If `transcribe_only` is True,
        it's wrapped in a dictionary with a `transcription_only` flag.
    """
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
    video_path: str,
    project_path: str,
    transcript_path: str = None,
    num_topics: int = 5,
    skip_unsilence: bool = False,
    transcribe_only: bool = False,
    is_youtube_url: bool = False,
    software_list: list = None,
    ocr_lang: str = "eng",
    frames_per_scene: int = 1,
    register: str = "it-workflow",
    progress_json: bool = False,
    min_segment_duration: float = 30.0,
    max_segment_duration: float = 300.0,
    topic_confidence_threshold: float = 0.7,
    preserve_natural_breaks: bool = True,
    topic_similarity_threshold: float = 0.6,
    max_merge_passes: int = 3,
) -> dict:
    """
    Execute the main video processing pipeline.

    This function coordinates the entire video analysis process, including:
    - Setting up project structure.
    - Handling YouTube downloads.
    - Processing audio and generating transcripts.
    - Performing topic modeling on the transcript.
    - Segmenting the video based on identified topics.
    - Conducting multimodal analysis on each segment.
    - Saving all results and checkpoints.

    Args:
        video_path: Path to the input video or YouTube URL.
        project_path: The root directory for all output files.
        transcript_path: Optional path to a pre-existing transcript file.
        num_topics: The number of topics to identify in the transcript.
        skip_unsilence: If True, skips the audio silence removal step.
        transcribe_only: If True, stops after generating the transcript.
        is_youtube_url: Flag indicating if the video_path is a YouTube URL.
        software_list: A list of software names to detect in video frames.
        ocr_lang: The language to use for Optical Character Recognition (OCR).
        frames_per_scene: The number of frames to analyze per video scene.
        register: The analysis register for tailoring AI analysis.
        progress_json: If True, outputs progress updates in JSON format.
        min_segment_duration: Minimum duration for merged segments in seconds.
        max_segment_duration: Maximum duration for merged segments in seconds.
        topic_confidence_threshold: Minimum confidence to merge segments.
        preserve_natural_breaks: Whether to respect natural pauses/breaks.

    Returns:
        A dictionary containing the comprehensive results of the analysis,
        including topics, segments, and paths to generated files.
    """
    from .project import load_checkpoint

    # Initialize progress tracker
    progress_tracker = ProgressTracker(project_path, progress_json)
    progress_tracker.add_callback(create_console_progress_callback())

    try:
        # Initialize project structure
        project_structure = ProjectStructure(project_path)
        project_structure.create_base_structure()

        # Move input files to organized structure
        input_files = project_structure.move_input_files(
            video_path, transcript_path)
        video_path = input_files["video"]  # Use the copied video file

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
                result = download_video(
                    video_path, download_path, project_path)
                if result["status"] == "error":
                    raise RuntimeError(
                        f"YouTube download failed: {result['message']}")
                video_path = download_path
                save_checkpoint(
                    project_path,
                    CHECKPOINTS["YOUTUBE_DOWNLOAD_COMPLETE"],
                    {"video_path": video_path,
                        "thumbnail_info": result.get("thumbnail_info")},
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
                unsilenced_video_path = checkpoint["data"].get(
                    "unsilenced_video_path", video_path)
                mono_resampled_audio_path = checkpoint["data"]["mono_resampled_audio_path"]

        if checkpoint is None or checkpoint["stage"] < CHECKPOINTS["PROCESS_COMPLETE"]:
            transcript = get_or_create_transcript(
                transcript_path, mono_resampled_audio_path, project_path, transcribe_only
            )
            if transcribe_only:
                return transcript

            # Topic modeling and transcript processing
            topic_results = process_transcript(
                transcript, 
                project_path, 
                num_topics, 
                register=register, 
                debug=False, 
                progress_tracker=progress_tracker,
                min_segment_duration=min_segment_duration,
                max_segment_duration=max_segment_duration,
                topic_confidence_threshold=topic_confidence_threshold,
                preserve_natural_breaks=preserve_natural_breaks,
                topic_similarity_threshold=topic_similarity_threshold,
                max_merge_passes=max_merge_passes
            )

            # Save transcript files in organized structure
            project_structure.save_transcript_files(transcript, topic_results)

            # NEW: Video segmentation based on topics
            if progress_tracker:
                progress_tracker.start_phase("Video Segmentation")

            topic_segments = topic_results.get("segments", [])
            segmented_files = segment_video_by_topics(
                unsilenced_video_path,
                topic_segments,
                project_structure,
                progress_tracker
            )

            # NEW: Segment-level multimodal analysis
            if progress_tracker:
                progress_tracker.start_phase("Segment Analysis")

            segment_processor = SegmentProcessor(progress_tracker)
            processed_segments = segment_processor.process_segments(
                unsilenced_video_path,
                segmented_files,
                transcript
            )

            # Save segment results
            segment_results = segment_processor.save_segment_results(
                processed_segments, project_structure
            )

            # Keep legacy visual analysis for backward compatibility
            analyzed_scenes = split_and_analyze_video(
                unsilenced_video_path,
                project_path,
                software_list,
                ocr_lang,
                frames_per_scene,
                register,
                progress_tracker,
            )

            # Combine results with new structure
            results = {
                "topics": topic_results.get("topics", []),
                "segments": topic_results.get("segments", []),
                "analyzed_scenes": analyzed_scenes,
                "segmented_files": segmented_files,
                "processed_segments": processed_segments,
                "segment_results": segment_results,
                "project_structure": {
                    "input_files": input_files,
                    "transcript_dir": project_structure.get_transcript_dir(),
                    "segments_dir": project_structure.get_topic_segments_dir(),
                    "final_analysis_dir": project_structure.get_final_analysis_dir()
                }
            }

            results_path = os.path.join(project_path, "results.json")
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

            # Clean up legacy files
            project_structure.migrate_legacy_files()

            save_checkpoint(project_path, CHECKPOINTS["PROCESS_COMPLETE"], {
                            "results": results})

            progress_tracker.start_phase("Process Complete")
            progress_tracker.update_phase_progress(
                100.0, "Processing complete - organized structure created")
            progress_tracker.complete_phase("Process Complete")
        else:
            results = checkpoint["data"]["results"]
            progress_tracker.start_phase("Process Complete")
            progress_tracker.update_phase_progress(
                100.0, "Using cached results")
            progress_tracker.complete_phase("Process Complete")

        return results

    except Exception as e:
        if progress_tracker:
            progress_tracker.fail_phase(f"Processing failed: {str(e)}")
        raise
