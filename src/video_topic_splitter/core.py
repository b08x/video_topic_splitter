#!/usr/bin/env python3
"""Core processing functionality for video topic splitter."""

import json
import os
import shutil  # Added for clarity as it's used conditionally

from deepgram import DeepgramClient, PrerecordedOptions
from dotenv import load_dotenv
from groq import Groq

from .analysis.topic_modeling import process_transcript
from .analysis.visual_analysis import split_and_analyze_video
from .api.deepgram import transcribe_file_deepgram
from .constants import CHECKPOINTS
from .processing.audio.audio import (convert_to_mono_and_resample,
                                     extract_audio, normalize_audio,
                                     remove_silence)
from .project import save_checkpoint, load_checkpoint # Added load_checkpoint for clarity
from .transcription import load_transcript, save_transcript, save_transcription
from .utils.youtube import download_video

load_dotenv()


def handle_audio_video(video_path, project_path, skip_unsilence=False):
    """
    Processes the audio track of a video file, including normalization, silence removal,
    extraction, and resampling, with checkpointing and file caching.

    This function performs several audio processing steps:
    1. Normalizes the audio levels in the video.
    2. Optionally removes silent sections from the video.
    3. Extracts the audio track from the (potentially unsilenced) video.
    4. Converts the extracted audio to mono and resamples it to a standard rate.

    It checks for existing processed files at each step to avoid redundant work.
    Checkpoints are saved upon successful completion of all steps.

    Args:
        video_path (str): The path to the input video file.
        project_path (str): The path to the project directory where processed files
                            and checkpoints will be saved.
        skip_unsilence (bool, optional): If True, skips the silence removal step.
                                         Defaults to False.

    Returns:
        tuple[str, str]: A tuple containing:
            - unsilenced_video_path (str): Path to the video file after normalization
                                           and optional silence removal.
            - mono_resampled_audio_path (str): Path to the final processed audio file
                                               (mono, resampled).

    Raises:
        RuntimeError: If audio normalization, silence removal (if attempted),
                      or audio conversion fails.
        FileNotFoundError: If the input `video_path` does not exist (implicitly raised
                           by underlying ffmpeg commands).
        Exception: Propagates exceptions from underlying audio processing functions
                   (e.g., `extract_audio`).

    Side Effects:
        - Creates an 'audio' subdirectory within `project_path` if it doesn't exist.
        - Creates intermediate and final processed video/audio files within `project_path`.
        - Saves a checkpoint file (`checkpoint.json`) in `project_path` upon successful
          completion.
    """
    audio_dir = os.path.join(project_path, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    # Extract file name and extension
    video_name, video_ext = os.path.splitext(os.path.basename(video_path))

    # Define paths using original file extension for video, specific extensions for audio
    normalized_video_path = os.path.join(project_path, f"normalized_video{video_ext}")
    unsilenced_video_path = os.path.join(project_path, f"unsilenced_video{video_ext}")
    raw_audio_path = os.path.join(audio_dir, "extracted_audio.opus") # Using opus for intermediate
    mono_resampled_audio_path = os.path.join(audio_dir, "mono_resampled_audio.m4a") # Using m4a for final

    # Check for existing final processed files first
    if os.path.exists(unsilenced_video_path) and os.path.exists(
        mono_resampled_audio_path
    ):
        print("Found existing processed audio/video files. Using cached versions.")
        return unsilenced_video_path, mono_resampled_audio_path

    # --- Normalization Step ---
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
    current_video_path = normalized_video_path # Keep track of the latest video version

    # --- Silence Removal Step ---
    if not os.path.exists(unsilenced_video_path):
        if skip_unsilence:
            # If skipping, the 'unsilenced' path is just a copy of the normalized one
            shutil.copy2(normalized_video_path, unsilenced_video_path)
            print("Skipping silence removal as requested. Using normalized video.")
        else:
            print("Removing silence...")
            silence_removal_result = remove_silence(
                normalized_video_path, unsilenced_video_path
            )
            if silence_removal_result["status"] == "error":
                print(
                    f"Error during silence removal: {silence_removal_result['message']}"
                )
                # Don't raise immediately, maybe extraction can still work from normalized
                print("Warning: Silence removal failed. Proceeding with normalized video for audio extraction.")
                # Fallback: use the normalized video if silence removal failed
                # We still need *an* unsilenced_video_path for the next step, even if it's just the normalized one
                if not os.path.exists(unsilenced_video_path):
                     shutil.copy2(normalized_video_path, unsilenced_video_path)

            else:
                print(silence_removal_result["message"])
    else:
         print("Using existing unsilenced video file.")
    # Update current video path regardless of success/failure/skip of silence removal
    current_video_path = unsilenced_video_path


    # --- Audio Extraction Step ---
    if not os.path.exists(raw_audio_path):
        print(f"Extracting audio from {os.path.basename(current_video_path)}...")
        try:
            # Use the potentially unsilenced video path for extraction
            extract_audio(current_video_path, raw_audio_path)
            print("Audio extraction complete.")
        except Exception as e:
            print(f"Error during audio extraction: {str(e)}")
            raise # Re-raise after logging
    else:
        print("Using existing extracted raw audio file.")


    # --- Conversion and Resampling Step ---
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

    # --- Final Check and Checkpointing ---
    # Ensure the final expected files exist before checkpointing
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
    else:
        # This case should ideally not be reached if errors were raised earlier,
        # but added as a safeguard.
        print("Warning: Final processed files not found. Checkpoint not saved.")
        # Depending on desired behavior, could raise an error here too.

    return unsilenced_video_path, mono_resampled_audio_path


def handle_transcription(
    video_path,
    audio_path,
    project_path,
    api="deepgram",
    num_topics=2,
    groq_prompt=None, # Keep for potential future use, but mark as unused currently
    software_list=None,
    logo_db_path=None,
    ocr_lang="eng",
    logo_threshold=0.8,
    thumbnail_interval=5,
    max_thumbnails=5,
    min_thumbnail_confidence=0.7,
    extract_scenes=False,
    min_scene_len=1.0,
    frames_per_scene=1,
    register="it-workflow",
):
    """
    Handles the transcription, topic modeling, and visual analysis stages.

    This function orchestrates several analysis steps:
    1. Transcribes the provided audio file using the specified API (currently Deepgram).
       Caches the full transcription response and a simplified transcript format.
    2. Loads an existing transcript if available.
    3. Performs topic modeling on the transcript to identify segments/chapters.
    4. Performs visual analysis on the video segments identified in step 3, including:
        - Scene detection (optional)
        - Frame extraction for thumbnails
        - OCR (Optical Character Recognition)
        - Logo detection
        - Software identification (based on visual cues)

    Checkpoints are saved after transcription and after successful video analysis.

    Args:
        video_path (str): Path to the video file (used for visual analysis). This should
                          typically be the `unsilenced_video_path` from `handle_audio_video`.
        audio_path (str): Path to the processed audio file (used for transcription). This
                          should typically be the `mono_resampled_audio_path` from
                          `handle_audio_video`.
        project_path (str): Path to the project directory for saving outputs and checkpoints.
        api (str, optional): The transcription API to use. Currently only "deepgram"
                             is implemented. Defaults to "deepgram".
        num_topics (int, optional): The target number of topics/segments for topic modeling.
                                    Defaults to 2.
        groq_prompt (str | None, optional): Prompt for Groq API (currently unused).
                                            Defaults to None.
        software_list (list[str] | None, optional): A list of software names to look for
                                                    during visual analysis. Defaults to None.
        logo_db_path (str | None, optional): Path to the logo database file for logo detection.
                                             Defaults to None.
        ocr_lang (str, optional): Language(s) for OCR processing (e.g., 'eng', 'eng+fra').
                                  Defaults to "eng".
        logo_threshold (float, optional): Confidence threshold for logo detection (0.0 to 1.0).
                                          Defaults to 0.8.
        thumbnail_interval (int, optional): Interval (in seconds) between potential thumbnail
                                            frames within a segment. Defaults to 5.
        max_thumbnails (int, optional): Maximum number of thumbnails to extract per segment.
                                        Defaults to 5.
        min_thumbnail_confidence (float, optional): Minimum confidence score for a frame
                                                    to be considered a good thumbnail.
                                                    Defaults to 0.7.
        extract_scenes (bool, optional): Whether to perform scene detection within segments.
                                         Defaults to False.
        min_scene_len (float, optional): Minimum length (in seconds) for a detected scene.
                                         Defaults to 1.0.
        frames_per_scene (int, optional): Number of representative frames to extract per
                                          detected scene. Defaults to 1.
        register (str, optional): Identifier for the analysis workflow being used (passed
                                  to downstream analysis functions). Defaults to "it-workflow".

    Returns:
        dict: A dictionary containing the analysis results, including:
              - 'transcript': List of transcript segments (content, start, end).
              - 'segments': List of topic segments identified by topic modeling.
              - 'analyzed_segments': List of segments enriched with visual analysis data
                                     (thumbnails, OCR, logos, software).
              - Potentially other keys added by `process_transcript`.

    Raises:
        ValueError: If an unsupported `api` is specified, or if required API keys
                    (DG_API_KEY, GROQ_API_KEY if api='groq') are not found in environment variables.
        FileNotFoundError: If `audio_path` or `video_path` does not exist (implicitly).
        Exception: Propagates exceptions from transcription API calls or analysis functions.

    Side Effects:
        - Creates a 'segments' subdirectory within `project_path`.
        - Saves transcription files (`full_transcription.json`, `transcript.json`) in `project_path`.
        - Saves analysis results (`results.json`) in `project_path`.
        - Saves checkpoint files (`checkpoint.json`) in `project_path`.
        - Creates image files (thumbnails, scene frames) within the 'segments' directory.
    """
    segments_dir = os.path.join(project_path, "segments")
    os.makedirs(segments_dir, exist_ok=True)

    # --- Transcription ---
    transcript_path = os.path.join(project_path, "transcript.json")
    full_transcription_path = os.path.join(project_path, "full_transcription.json")
    transcript = None
    if os.path.exists(transcript_path):
        print("Loading existing simplified transcript...")
        transcript = load_transcript(transcript_path)
        print("Transcript loaded.")

    if not transcript:
        print("No transcript found or failed to load. Transcribing audio...")
        deepgram_key = os.getenv("DG_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY") # Check even if unused for now

        if not deepgram_key:
            raise ValueError("DG_API_KEY environment variable is not set")
        # Note: Groq is not implemented here, but check kept for consistency
        if not groq_key and api == "groq":
            raise ValueError("GROQ_API_KEY environment variable is not set for Groq API")

        if api == "deepgram":
            print(f"Using Deepgram API with audio: {audio_path}")
            deepgram_client = DeepgramClient(deepgram_key)
            # Define Deepgram options for comprehensive analysis
            deepgram_options = PrerecordedOptions(
                model="nova-2",          # Recommended model
                language="en",           # Assuming English, could be parameterized
                topics=True,             # Request topic segments from Deepgram
                intents=True,            # Request intent detection
                smart_format=True,       # Apply formatting (punctuation, etc.)
                punctuate=True,          # Ensure punctuation
                paragraphs=True,         # Group into paragraphs
                utterances=True,         # Get utterance-level timestamps
                diarize=True,            # Attempt speaker diarization
                filler_words=True,       # Include filler words (e.g., "um")
                sentiment=True,          # Analyze sentiment
                # Consider adding summarization=True if needed
            )
            # Perform transcription
            transcription = transcribe_file_deepgram(
                deepgram_client, audio_path, deepgram_options
            )

            # Save the full response
            save_transcription(transcription, project_path) # Saves to full_transcription.json

            # Extract simplified transcript (utterances) for topic modeling
            if transcription and "results" in transcription and "utterances" in transcription["results"]:
                 transcript = [
                    {
                        "content": utterance["transcript"],
                        "start": utterance["start"],
                        "end": utterance["end"],
                        # Optionally add speaker info if diarization is reliable
                        # "speaker": utterance.get("speaker", "UNKNOWN")
                    }
                    for utterance in transcription["results"]["utterances"]
                 ]
                 # Save the simplified transcript
                 save_transcript(transcript, project_path) # Saves to transcript.json
            else:
                 print("Warning: Transcription response did not contain expected 'results.utterances'. Cannot create simplified transcript.")
                 # Decide how to handle this - raise error or proceed without transcript?
                 # For now, let it proceed, topic modeling will likely fail gracefully.
                 transcript = [] # Ensure transcript is an empty list if extraction fails

        # Placeholder for other APIs
        # elif api == "groq":
        #     print(f"Using Groq API with audio: {audio_path}")
        #     # ... Groq transcription logic would go here ...
        #     raise NotImplementedError("Groq transcription is not yet implemented.")
        else:
            # Should not be reachable if using default 'deepgram' but good practice
            raise ValueError(
                f"API '{api}' is not currently supported."
            )

    # Checkpoint after transcription attempt (even if loading from cache)
    if transcript is not None: # Check if transcript exists (loaded or generated)
        save_checkpoint(
            project_path, CHECKPOINTS["TRANSCRIPTION_COMPLETE"], {"transcript_path": transcript_path}
        )
        print("Transcription checkpoint saved.")
    else:
        # This indicates a failure to load *or* generate a transcript.
        # Depending on requirements, might want to raise an error here.
        print("Warning: No transcript available after transcription step.")
        # Create an empty results dict to return if subsequent steps fail
        results = {"transcript": [], "analyzed_segments": [], "error": "Transcript generation failed"}


    # --- Topic Modeling ---
    # Proceed only if a transcript (even an empty one) is available
    if transcript is not None:
        print("Processing transcript for topic modeling...")
        # `process_transcript` should handle empty transcripts gracefully
        results = process_transcript(
            transcript, project_path, num_topics, register=register
        )
        print(f"Topic modeling complete. Found {len(results.get('segments', []))} potential segments.")
    else:
        # Skip topic modeling if transcript failed
        print("Skipping topic modeling due to missing transcript.")
        # Ensure results dict exists with empty segments if transcript was None
        if 'results' not in locals():
             results = {"transcript": [], "segments": [], "analyzed_segments": [], "error": "Transcript generation failed"}
        else: # Add empty segments if transcript existed but processing failed before this point
             results["segments"] = []


    # --- Visual Analysis ---
    # Proceed only if topic modeling produced segments (or if we want to analyze the whole video)
    # Check if 'segments' key exists and is not empty
    if "segments" in results and results["segments"]:
        print(f"Starting visual analysis for {len(results['segments'])} segments...")
        try:
            analyzed_segments = split_and_analyze_video(
                video_path=video_path,
                segments=results["segments"], # Pass the segments from topic modeling
                output_dir=segments_dir,
                software_list=software_list,
                logo_db_path=logo_db_path,
                ocr_lang=ocr_lang,
                logo_threshold=logo_threshold,
                thumbnail_interval=thumbnail_interval,
                max_thumbnails=max_thumbnails,
                min_thumbnail_confidence=min_thumbnail_confidence,
                extract_scenes=extract_scenes,
                min_scene_len=min_scene_len,
                frames_per_scene=frames_per_scene,
                register=register,
            )

            # Update results with analyzed segments
            results["analyzed_segments"] = analyzed_segments
            print(f"Successfully analyzed {len(analyzed_segments)} segments visually.")

        except Exception as e:
            print(f"Error during video analysis: {str(e)}")
            # Attempt to load any segments that might have been partially processed before the error
            # Note: The current split_and_analyze_video might not support partial recovery easily.
            # This recovery logic might need refinement based on how split_and_analyze_video handles errors.
            print("Attempting to recover partially analyzed segments...")
            try:
                # Calling with empty segments might just list existing segment folders/files
                # This behavior depends heavily on the implementation of split_and_analyze_video
                recovered_segments = split_and_analyze_video(video_path, [], segments_dir) # Check if this works for recovery
                if recovered_segments:
                    results["analyzed_segments"] = recovered_segments
                    print(f"Recovered {len(recovered_segments)} previously analyzed segments.")
                else:
                    results["analyzed_segments"] = []
                    print("No previously analyzed segments found or recovered.")
            except Exception as load_error:
                print(f"Could not load/recover analyzed segments: {str(load_error)}")
                results["analyzed_segments"] = [] # Ensure it's an empty list on failure
    else:
        print("Skipping visual analysis as no segments were generated from topic modeling.")
        results["analyzed_segments"] = [] # Ensure key exists even if skipped


    # --- Save Final Results ---
    results_path = os.path.join(project_path, "results.json")
    print(f"Saving final analysis results to {results_path}")
    try:
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print("Results saved successfully.")
    except Exception as e:
        print(f"Error saving results.json: {str(e)}")
        # Decide if this should be a critical error

    # Checkpoint after visual analysis attempt
    save_checkpoint(project_path, CHECKPOINTS["VIDEO_ANALYZED"], {"results_path": results_path})
    print("Video analysis checkpoint saved.")

    return results


def process_video(
    video_path,
    project_path,
    api="deepgram",
    num_topics=2,
    groq_prompt=None, # Keep for potential future use
    skip_unsilence=False,
    transcribe_only=False,
    is_youtube_url=False,
    software_list=None,
    logo_db_path=None,
    ocr_lang="eng",
    logo_threshold=0.8,
    thumbnail_interval=5,
    max_thumbnails=5,
    min_thumbnail_confidence=0.7,
    extract_scenes=False,
    min_scene_len=1.0,
    frames_per_scene=1,
    register="it-workflow",
):
    """
    Main pipeline function to process a video file or YouTube URL.

    This function coordinates the entire video processing workflow:
    1. Handles optional YouTube video downloading.
    2. Processes the video's audio track (normalization, silence removal, extraction, resampling)
       via `handle_audio_video`.
    3. If `transcribe_only` is False:
        - Performs transcription, topic modeling, and visual analysis via `handle_transcription`.
    4. If `transcribe_only` is True:
        - Performs only transcription using the specified API.
        - Skips topic modeling and visual analysis.
    5. Uses checkpointing to resume processing from the last completed stage.

    Args:
        video_path (str): The path to the local video file or the YouTube URL.
        project_path (str): The path to the project directory where all outputs,
                            intermediate files, and checkpoints will be stored.
                            This directory will be created if it doesn't exist.
        api (str, optional): The transcription API to use ("deepgram"). Defaults to "deepgram".
        num_topics (int, optional): Target number of topics for topic modeling. Defaults to 2.
                                    (Used only if `transcribe_only` is False).
        groq_prompt (str | None, optional): Prompt for Groq API (currently unused).
                                            Defaults to None.
        skip_unsilence (bool, optional): If True, skips the silence removal step during
                                         audio processing. Defaults to False.
        transcribe_only (bool, optional): If True, only performs audio extraction and
                                          transcription, skipping topic modeling and
                                          visual analysis. Defaults to False.
        is_youtube_url (bool, optional): Set to True if `video_path` is a YouTube URL.
                                         Defaults to False.
        software_list (list[str] | None, optional): List of software names for visual analysis.
                                                    Defaults to None. (Used only if `transcribe_only` is False).
        logo_db_path (str | None, optional): Path to logo database for visual analysis.
                                             Defaults to None. (Used only if `transcribe_only` is False).
        ocr_lang (str, optional): Language(s) for OCR. Defaults to "eng".
                                  (Used only if `transcribe_only` is False).
        logo_threshold (float, optional): Confidence threshold for logo detection. Defaults to 0.8.
                                          (Used only if `transcribe_only` is False).
        thumbnail_interval (int, optional): Interval (seconds) for thumbnail candidates. Defaults to 5.
                                            (Used only if `transcribe_only` is False).
        max_thumbnails (int, optional): Max thumbnails per segment. Defaults to 5.
                                        (Used only if `transcribe_only` is False).
        min_thumbnail_confidence (float, optional): Min confidence for good thumbnails. Defaults to 0.7.
                                                    (Used only if `transcribe_only` is False).
        extract_scenes (bool, optional): Whether to perform scene detection. Defaults to False.
                                         (Used only if `transcribe_only` is False).
        min_scene_len (float, optional): Minimum scene length (seconds). Defaults to 1.0.
                                         (Used only if `transcribe_only` is False).
        frames_per_scene (int, optional): Frames to extract per scene. Defaults to 1.
                                          (Used only if `transcribe_only` is False).
        register (str, optional): Identifier for the analysis workflow. Defaults to "it-workflow".
                                  (Used only if `transcribe_only` is False).

    Returns:
        dict: A dictionary containing the final results.
              If `transcribe_only` is True, it contains {'transcript': ..., 'transcription_only': True}.
              If `transcribe_only` is False, it contains the full analysis results from
              `handle_transcription`.

    Raises:
        RuntimeError: If YouTube download fails or if critical errors occur during
                      audio processing (`handle_audio_video`).
        ValueError: If required API keys are missing or an invalid API is specified.
        FileNotFoundError: If input files are not found at various stages (implicitly).
        Exception: Propagates exceptions from underlying functions.

    Side Effects:
        - Creates the `project_path` directory if it doesn't exist.
        - Creates subdirectories ('audio', 'segments') and numerous files within `project_path`.
        - Saves checkpoint files (`checkpoint.json`) at various stages.
        - Prints status messages to the console.
    """
    # Ensure project path exists
    os.makedirs(project_path, exist_ok=True)

    # Load the last checkpoint, if any
    checkpoint = load_checkpoint(project_path)
    current_stage = checkpoint["stage"] if checkpoint else -1 # Use -1 if no checkpoint

    # --- Stage 1: YouTube Download (if applicable) ---
    if is_youtube_url:
        youtube_complete_stage = CHECKPOINTS["YOUTUBE_DOWNLOAD_COMPLETE"]
        if current_stage < youtube_complete_stage:
            print("Downloading YouTube video...")
            # Define download path within the project directory
            download_path = os.path.join(project_path, "source_video.mp4") # Standardize name
            result = download_video(video_path, download_path, project_path) # Pass project_path for potential metadata

            if result["status"] == "error":
                raise RuntimeError(f"YouTube download failed: {result['message']}")

            video_path = download_path # Update video_path to the downloaded file
            save_checkpoint(
                project_path,
                youtube_complete_stage,
                {
                    "video_path": video_path, # Save the path to the downloaded video
                    "thumbnail_info": result.get("thumbnail_info"), # Store thumbnail info if available
                    "original_url": video_path # Keep track of the original URL maybe?
                },
            )
            print(result["message"])
            current_stage = youtube_complete_stage # Update current stage
        else:
            # Load video path from checkpoint if download was already done
            video_path = checkpoint["data"]["video_path"]
            print(f"Using previously downloaded YouTube video: {video_path}")
    # If not a YouTube URL, video_path remains the initial local path


    # --- Stage 2: Audio Processing ---
    audio_processed_stage = CHECKPOINTS["AUDIO_PROCESSED"]
    if current_stage < audio_processed_stage:
        print("Starting audio processing...")
        # Ensure video_path points to the correct file (downloaded or local)
        if not os.path.exists(video_path):
             raise FileNotFoundError(f"Video file not found at expected path: {video_path}. Check previous steps or input.")

        unsilenced_video_path, mono_resampled_audio_path = handle_audio_video(
            video_path, project_path, skip_unsilence
        )
        # handle_audio_video saves its own checkpoint upon success
        current_stage = audio_processed_stage # Update stage after successful completion
    else:
        # Load paths from the checkpoint data
        unsilenced_video_path = checkpoint["data"]["unsilenced_video_path"]
        mono_resampled_audio_path = checkpoint["data"]["mono_resampled_audio_path"]
        print("Audio processing already completed. Using cached paths.")
        print(f"  Unsilenced Video: {unsilenced_video_path}")
        print(f"  Mono Audio: {mono_resampled_audio_path}")


    # --- Stage 3: Transcription and Analysis (or Transcription Only) ---
    # Determine the target final stage based on transcribe_only flag
    analysis_complete_stage = CHECKPOINTS["VIDEO_ANALYZED"]
    transcribe_only_stage = CHECKPOINTS["TRANSCRIBE_ONLY_COMPLETE"]
    target_stage = transcribe_only_stage if transcribe_only else analysis_complete_stage

    results = None # Initialize results

    if current_stage < target_stage:
        if transcribe_only:
            print("Transcribe-only mode: Starting transcription...")
            # --- Transcription Only Logic ---
            transcript_path = os.path.join(project_path, "transcript.json")
            transcript = None
            if os.path.exists(transcript_path):
                try:
                    print("Loading existing transcript for transcribe-only mode...")
                    transcript = load_transcript(transcript_path)
                    print("Transcript loaded.")
                except Exception as e:
                    print(f"Warning: Failed to load existing transcript: {e}. Will re-transcribe.")
                    transcript = None # Ensure transcription happens if loading fails

            if transcript is None: # Proceed only if transcript wasn't loaded
                print("No transcript found or loading failed. Transcribing audio...")
                deepgram_key = os.getenv("DG_API_KEY")
                # No need to check GROQ key here as only Deepgram is implemented

                if not deepgram_key:
                    raise ValueError("DG_API_KEY environment variable is not set")

                if api == "deepgram":
                    deepgram_client = DeepgramClient(deepgram_key)
                    # Use simpler options for transcribe-only if full analysis isn't needed
                    deepgram_options = PrerecordedOptions(
                        model="nova-2",
                        language="en",
                        smart_format=True,
                        punctuate=True,
                        paragraphs=True, # Still useful for readability
                        utterances=True, # Essential for the desired output format
                        # Diarize, topics, intents etc., are likely not needed here
                    )
                    print(f"Transcribing audio file: {mono_resampled_audio_path}")
                    transcription = transcribe_file_deepgram(
                        deepgram_client, mono_resampled_audio_path, deepgram_options
                    )

                    # Save full response (optional but good for debugging)
                    save_transcription(transcription, project_path)

                    # Extract simplified transcript
                    if transcription and "results" in transcription and "utterances" in transcription["results"]:
                        transcript = [
                            {
                                "content": utterance["transcript"],
                                "start": utterance["start"],
                                "end": utterance["end"],
                            }
                            for utterance in transcription["results"]["utterances"]
                        ]
                        save_transcript(transcript, project_path) # Save the simplified version
                        print("Transcription complete and saved.")
                    else:
                         print("Warning: Transcription response did not contain expected 'results.utterances'.")
                         transcript = [] # Set to empty list on failure
                else:
                    # This path shouldn't be hit with default 'deepgram' but is a safeguard
                    raise ValueError(
                        f"API '{api}' is not currently supported for transcription."
                    )

            # Prepare results for transcribe-only mode
            results = {"transcript": transcript, "transcription_only": True}
            # Save checkpoint for transcribe-only completion
            save_checkpoint(
                project_path,
                transcribe_only_stage,
                {"results": results}, # Save the minimal results
            )
            print("Transcribe-only process complete.")
            current_stage = transcribe_only_stage

        else: # Full analysis mode
            print("Starting full transcription and analysis...")
            # --- Full Analysis Logic ---
            # Ensure the necessary input files exist before calling handle_transcription
            if not os.path.exists(unsilenced_video_path):
                 raise FileNotFoundError(f"Unsilenced video not found: {unsilenced_video_path}")
            if not os.path.exists(mono_resampled_audio_path):
                 raise FileNotFoundError(f"Mono audio not found: {mono_resampled_audio_path}")

            results = handle_transcription(
                video_path=unsilenced_video_path, # Use the processed video
                audio_path=mono_resampled_audio_path, # Use the processed audio
                project_path=project_path,
                api=api,
                num_topics=num_topics,
                groq_prompt=groq_prompt, # Pass along even if unused by Deepgram path
                software_list=software_list,
                logo_db_path=logo_db_path,
                ocr_lang=ocr_lang,
                logo_threshold=logo_threshold,
                thumbnail_interval=thumbnail_interval,
                max_thumbnails=max_thumbnails,
                min_thumbnail_confidence=min_thumbnail_confidence,
                extract_scenes=extract_scenes,
                min_scene_len=min_scene_len,
                frames_per_scene=frames_per_scene,
                register=register,
            )
            # handle_transcription saves its own checkpoints internally
            # Check if the final results file was created successfully by handle_transcription
            results_path = os.path.join(project_path, "results.json")
            if os.path.exists(results_path):
                 current_stage = analysis_complete_stage # Update stage only on success
                 print("Full transcription and analysis process complete.")
            else:
                 print("Warning: Full analysis completed but results.json not found. Check handle_transcription logs.")
                 # Decide if this should prevent the final checkpoint

    else: # target_stage was already met or exceeded
        print("Transcription and Analysis stages already completed. Loading final results...")
        # Load results from the last relevant checkpoint
        if checkpoint and "results" in checkpoint["data"]:
             results = checkpoint["data"]["results"]
             print("Loaded results from checkpoint.")
        elif os.path.exists(os.path.join(project_path, "results.json")):
             # Fallback: try loading results.json directly if checkpoint is missing results data
             try:
                 with open(os.path.join(project_path, "results.json"), 'r') as f:
                     results = json.load(f)
                 print("Loaded results from results.json file.")
             except Exception as e:
                 print(f"Error loading results.json: {e}. Cannot provide results.")
                 # What should be returned here? Maybe raise an error or return None/empty dict?
                 return {"error": "Failed to load previous results."}
        else:
             print("Warning: Could not find previous results in checkpoint or results.json.")
             # Decide return value in this edge case
             return {"error": "Previous results not found."}


    # --- Final Checkpoint ---
    # Save a final overall completion checkpoint, regardless of mode
    final_checkpoint_stage = (
        transcribe_only_stage if transcribe_only else CHECKPOINTS["PROCESS_COMPLETE"] # Use a dedicated final stage
    )
    # Only save if results were successfully obtained or loaded
    if results is not None:
        save_checkpoint(project_path, final_checkpoint_stage, {"results": results})
        print(f"Final process checkpoint saved (Stage: {final_checkpoint_stage}).")
    else:
        print("Warning: Final results are missing, skipping final checkpoint.")


    return results

