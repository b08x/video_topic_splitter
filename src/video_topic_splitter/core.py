#!/usr/bin/env python3
"""Core processing functionality for video scene splitter."""

import json
import logging
import os
import shutil
import csv
from typing import Optional, List, Dict, Any, Tuple

from deepgram import DeepgramClient, PrerecordedOptions
from dotenv import load_dotenv

# Local application imports
from .constants import CHECKPOINTS, NO_SCENES_DETECTED
from .project import load_checkpoint, save_checkpoint
from .utils.youtube import download_video, is_youtube_url
from .processing.audio.audio import (convert_to_mono_and_resample,
                                     extract_audio, normalize_audio,
                                     remove_silence)
from .processing.video.scene_detection import (detect_scenes,
                                               split_video_by_scenes) # Keep for splitting
# For visual analysis and multimodal topic modeling
from .analysis.visual_analysis import analyze_scenes
from .analysis.visual_topic_modeling import prepare_visual_frames_for_topic_modeling, process_transcript_with_visuals
from .api.gemini import GeminiClient
from .api.deepgram import transcribe_file_deepgram
from .transcription import load_transcript, save_transcript, save_transcription


load_dotenv()
logger = logging.getLogger(__name__)

# Note: handle_audio_video function remains largely the same as before
# It processes audio and prepares it for transcription.
def handle_audio_video(video_path, project_path, skip_unsilence=False):
    """
    Processes the audio track of a video file: normalization, optional silence
    removal, extraction, and resampling, with checkpointing.
    (Implementation details omitted for brevity - assume it's the same as previously shown)
    """
    audio_dir = os.path.join(project_path, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    logger.info(f"Audio processing outputs will be saved in: {audio_dir}")

    # Checkpoint: PROJECT_CREATED implicitly handled by project folder creation

    video_name, video_ext = os.path.splitext(os.path.basename(video_path))
    normalized_video_path = os.path.join(project_path, f"normalized_video{video_ext}")
    unsilenced_video_path = os.path.join(project_path, f"unsilenced_video{video_ext}")
    raw_audio_path = os.path.join(audio_dir, "extracted_audio.opus")
    mono_resampled_audio_path = os.path.join(audio_dir, "mono_resampled_audio.m4a")

    # Check for existing final processed files first
    if os.path.exists(unsilenced_video_path) and os.path.exists(mono_resampled_audio_path):
        logger.info("Found existing processed audio/video files. Using cached versions.")
        # Ensure checkpoint reflects this if loading from cache
        save_checkpoint(
            project_path,
            CHECKPOINTS["AUDIO_PROCESSED"],
            {
                "processed_video_path": unsilenced_video_path,
                "processed_audio_path": mono_resampled_audio_path,
            },
        )
        return unsilenced_video_path, mono_resampled_audio_path

    current_video_path = video_path # Start with original video

    # --- Normalization Step ---
    if not os.path.exists(normalized_video_path):
        logger.info("Normalizing audio...")
        normalize_result = normalize_audio(current_video_path, normalized_video_path)
        if normalize_result["status"] == "error":
            logger.error(f"Audio normalization failed: {normalize_result['message']}")
            raise RuntimeError("Audio normalization failed")
        logger.info("Audio normalization complete.")
        current_video_path = normalized_video_path # Update path
    else:
        logger.info("Using existing normalized video file.")
        current_video_path = normalized_video_path

    # --- Silence Removal Step ---
    if not os.path.exists(unsilenced_video_path):
        if skip_unsilence:
            logger.info("Skipping silence removal as requested.")
            try:
                shutil.copy2(current_video_path, unsilenced_video_path)
                logger.info(f"Copied {os.path.basename(current_video_path)} to {os.path.basename(unsilenced_video_path)}")
            except Exception as e:
                 logger.error(f"Failed to copy normalized video for skipping unsilence: {e}")
                 raise RuntimeError("Failed to prepare video for skipping unsilence step.") from e
        else:
            logger.info("Removing silence...")
            silence_removal_result = remove_silence(
                current_video_path, unsilenced_video_path
            )
            if silence_removal_result["status"] == "error":
                logger.warning(f"Silence removal failed: {silence_removal_result['message']}. Proceeding with normalized video.")
                # Fallback: copy normalized if unsilence failed to create output
                if not os.path.exists(unsilenced_video_path):
                     try:
                        shutil.copy2(current_video_path, unsilenced_video_path)
                        logger.info(f"Using normalized video as fallback: {os.path.basename(unsilenced_video_path)}")
                     except Exception as e:
                         logger.error(f"Failed to copy normalized video as fallback: {e}")
                         raise RuntimeError("Silence removal failed and fallback copy also failed.") from e
            else:
                logger.info("Silence removal complete.")
    else:
         logger.info("Using existing unsilenced video file.")

    # Update current video path to the one that will be used for extraction
    current_video_path_for_extraction = unsilenced_video_path


    # --- Audio Extraction Step ---
    if not os.path.exists(raw_audio_path):
        logger.info(f"Extracting audio from {os.path.basename(current_video_path_for_extraction)}...")
        try:
            extract_audio(current_video_path_for_extraction, raw_audio_path)
            logger.info("Audio extraction complete.")
        except Exception as e:
            logger.error(f"Audio extraction failed: {str(e)}")
            raise # Re-raise critical error
    else:
        logger.info("Using existing extracted raw audio file.")

    # --- Conversion and Resampling Step ---
    if not os.path.exists(mono_resampled_audio_path):
        logger.info("Converting audio to mono and resampling...")
        conversion_result = convert_to_mono_and_resample(
            raw_audio_path, mono_resampled_audio_path
        )
        if conversion_result["status"] == "error":
            logger.error(f"Audio conversion failed: {conversion_result['message']}")
            raise RuntimeError("Audio conversion failed")
        logger.info("Audio conversion and resampling complete.")
    else:
        logger.info("Using existing mono resampled audio file.")

    # --- Final Check and Checkpointing ---
    if os.path.exists(unsilenced_video_path) and os.path.exists(mono_resampled_audio_path):
        save_checkpoint(
            project_path,
            CHECKPOINTS["AUDIO_PROCESSED"],
            {
                "processed_video_path": unsilenced_video_path,
                "processed_audio_path": mono_resampled_audio_path,
            },
        )
        logger.info("Audio processing checkpoint saved.")
    else:
        # This case should ideally not be reached if checks above passed, but added for safety
        logger.error("Final processed audio/video files not found after processing steps. Checkpoint not saved.")
        raise RuntimeError("Audio processing finished but expected output files are missing.")

    return unsilenced_video_path, mono_resampled_audio_path


# Note: handle_transcription_and_scene_detection remains largely the same
# It handles transcription and scene detection concurrently/sequentially.
def handle_transcription_and_scene_detection(
    audio_path: str,
    video_path: str, # Needed for scene detection
    project_path: str,
    api: str = "deepgram",
    scene_threshold: float = 27.0,
    min_scene_len_sec: float = 1.0,
) -> tuple[Optional[list], Optional[list]]:
    """
    Handles audio transcription and video scene detection.
    (Implementation details omitted for brevity - assume it's the same as previously shown)
    Saves checkpoints: TRANSCRIPTION_COMPLETE, SCENES_DETECTED, NO_SCENES_DETECTED
    """
    # --- Transcription ---
    transcript_path = os.path.join(project_path, "transcript.json")
    full_transcription_path = os.path.join(project_path, "full_transcription.json")
    transcript = None
    if os.path.exists(transcript_path):
        logger.info("Loading existing simplified transcript...")
        try:
            transcript = load_transcript(transcript_path)
            logger.info("Transcript loaded.")
        except Exception as e:
             logger.warning(f"Failed to load existing transcript ({transcript_path}): {e}. Will re-transcribe.")
             transcript = None # Ensure re-transcription

    if transcript is None:
        logger.info("Transcribing audio...")
        deepgram_key = os.getenv("DG_API_KEY")
        if not deepgram_key:
            raise ValueError("DG_API_KEY environment variable is not set")

        if api == "deepgram":
            try:
                deepgram_client = DeepgramClient(deepgram_key)
                deepgram_options = PrerecordedOptions(
                    model="nova-2", language="en", smart_format=True,
                    punctuate=True, utterances=True,
                )
                transcription = transcribe_file_deepgram(
                    deepgram_client, audio_path, deepgram_options
                )
                save_transcription(transcription, project_path) # Save full response

                # Extract simplified transcript
                if transcription and "results" in transcription and "utterances" in transcription["results"]:
                    transcript = [
                        {"content": utt["transcript"], "start": utt["start"], "end": utt["end"]}
                        for utt in transcription["results"]["utterances"]
                    ]
                    save_transcript(transcript, project_path) # Save simplified version
                    logger.info("Transcription complete and saved.")
                else:
                    logger.error("Transcription response missing 'results.utterances'.")
                    transcript = None # Indicate failure
            except Exception as e:
                 logger.error(f"Deepgram transcription failed: {e}", exc_info=True)
                 transcript = None # Indicate failure
        else:
            # Placeholder for other APIs if needed
            raise ValueError(f"Transcription API '{api}' is not currently supported.")

    # Checkpoint after transcription attempt
    if transcript is not None:
        save_checkpoint(
            project_path, CHECKPOINTS["TRANSCRIPTION_COMPLETE"], {"transcript_path": transcript_path}
        )
        logger.info("Transcription checkpoint saved.")
    else:
        logger.error("Transcription failed or produced no utterances.")
        # Return None for transcript, main function should handle this

    # --- Scene Detection ---
    scene_boundaries = None
    scenes_csv_path = os.path.join(project_path, "scenes", "scenes.csv") # Standard path for CSV
    if os.path.exists(scenes_csv_path):
        logger.info("Loading existing scene boundaries from CSV...")
        try:
             scene_boundaries = []
             with open(scenes_csv_path, 'r', encoding='utf-8') as f:
                 reader = csv.reader(f)
                 header = next(reader) # Skip header
                 for row in reader:
                     try:
                         start_sec = float(row[3])
                         end_sec = float(row[4])
                         scene_boundaries.append((start_sec, end_sec))
                     except (IndexError, ValueError) as row_err:
                         logger.warning(f"Skipping invalid row in scenes.csv: {row} - {row_err}")
             logger.info(f"Loaded {len(scene_boundaries)} scenes from CSV.")
        except Exception as e:
             logger.warning(f"Failed to load scenes from CSV ({scenes_csv_path}): {e}. Will re-detect.")
             scene_boundaries = None # Ensure re-detection

    if scene_boundaries is None:
        logger.info("Detecting scenes in video...")
        scenes_output_dir = os.path.join(project_path, "scenes") # Dir for CSV and maybe frames later
        os.makedirs(scenes_output_dir, exist_ok=True) # Ensure dir exists
        try:
            scene_boundaries = detect_scenes(
                video_path=video_path,
                output_dir=scenes_output_dir,
                threshold=scene_threshold,
                min_scene_len_sec=min_scene_len_sec,
                save_csv=True # Ensure CSV is saved
            )

            if scene_boundaries and len(scene_boundaries) > 0:
                logger.info(f"Scene detection complete. Found {len(scene_boundaries)} scenes.")
                save_checkpoint(
                    project_path, CHECKPOINTS["SCENES_DETECTED"], {"scene_boundaries": scene_boundaries}
                )
                logger.info("Scene detection checkpoint saved.")
            else:
                logger.warning("No scenes were detected in the video.")
                save_checkpoint(project_path, NO_SCENES_DETECTED, {"message": "No scenes detected in video"})
                scene_boundaries = [] # Use empty list for consistency
        except Exception as e:
            logger.error(f"Scene detection failed: {e}", exc_info=True)
            scene_boundaries = None # Indicate failure

    # Return results
    return transcript, scene_boundaries


# --- Main Processing Function (Refactored) ---
def process_video(
    input_path: str, # Can be local path or YouTube URL
    project_path: str,
    api: str = "deepgram",
    skip_unsilence: bool = False,
    # Scene Detection Params
    scene_threshold: float = 27.0,
    min_scene_len: float = 1.0,
    # Visual Analysis Params passed down from CLI
    software_list: Optional[list] = None,
    ocr_lang: str = "eng",
    frames_per_scene: int = 1,
    frame_format: str = "jpg",
    compression_quality: int = 90,
    register: str = "it-workflow", # Default register
    visual_similarity_threshold: float = 0.6, # Default for VisualTopicAnalyzer
) -> Dict:
    # Initialize transcript_path at the beginning of the function
    transcript_path = os.path.join(project_path, "transcript.json")
    """
    Main unified pipeline function to process a video: download (optional),
    process audio, transcribe, detect scenes, perform visual analysis,
    run visual topic modeling, and split video based on multimodal segments.

    Args:
        input_path (str): Path to the local video file or YouTube URL.
        project_path (str): Path to the project directory.
        api (str, optional): Transcription API ('deepgram'). Defaults to "deepgram".
        skip_unsilence (bool, optional): Skip silence removal. Defaults to False.
        scene_threshold (float, optional): Threshold for scene detection. Defaults to 27.0.
        min_scene_len (float, optional): Min scene length (sec). Defaults to 1.0.
        software_list (list | None, optional): List of software names for OCR. Defaults to None.
        ocr_lang (str, optional): Language for OCR. Defaults to "eng".
        frames_per_scene (int, optional): Frames to extract/analyze per scene. Defaults to 1.
        frame_format (str, optional): Format for extracted frames. Defaults to "jpg".
        compression_quality (int, optional): JPEG quality. Defaults to 90.
        register (str, optional): Analysis register for context. Defaults to "it-workflow".
        visual_similarity_threshold (float, optional): Threshold for visual similarity detection. Defaults to 0.6.

    Returns:
        dict: Final results including transcript, visual topic analysis, and split video paths.

    Raises:
        RuntimeError, ValueError, FileNotFoundError, Exception: Propagated from sub-functions.
    """
    os.makedirs(project_path, exist_ok=True)
    logger.info(f"Starting unified processing pipeline for input: {input_path}")
    logger.info(f"Project directory: {project_path}")

    checkpoint = load_checkpoint(project_path)
    current_stage = checkpoint["stage"] if checkpoint else -1
    logger.info(f"Current checkpoint stage: {current_stage} (Using updated sequence)")

    is_youtube = is_youtube_url(input_path)
    video_path = input_path # May be updated after download

    # --- Stage 1: YouTube Download (if applicable) ---
    youtube_complete_stage = CHECKPOINTS["YOUTUBE_DOWNLOAD_COMPLETE"]
    if is_youtube and current_stage < youtube_complete_stage:
        logger.info("Downloading YouTube video...")
        download_path = os.path.join(project_path, "source_video.mp4") # Standard name
        result = download_video(input_path, download_path, project_path)
        if result["status"] == "error":
            raise RuntimeError(f"YouTube download failed: {result['message']}")
        video_path = result["file_path"] # Use the actual downloaded path
        save_checkpoint(project_path, youtube_complete_stage, {"video_path": video_path})
        logger.info(f"YouTube video downloaded to: {video_path}")
        current_stage = youtube_complete_stage
    elif is_youtube:
        # Ensure video_path is loaded from checkpoint if download was done previously
        if checkpoint and "video_path" in checkpoint["data"]:
             video_path = checkpoint["data"]["video_path"]
             logger.info(f"Using previously downloaded YouTube video: {video_path}")
        else:
             # Handle missing checkpoint data case
             raise RuntimeError("YouTube download checkpoint missing video path. Please clear checkpoint or re-run.")
    elif not os.path.exists(video_path):
         # If it's not YouTube and doesn't exist locally
         raise FileNotFoundError(f"Input video file not found: {video_path}")
    logger.info(f"Using video source: {video_path}")


    # --- Stage 2: Audio Processing ---
    audio_processed_stage = CHECKPOINTS["AUDIO_PROCESSED"]
    processed_video_path = None
    processed_audio_path = None
    if current_stage < audio_processed_stage:
        logger.info("Starting audio processing...")
        processed_video_path, processed_audio_path = handle_audio_video(
            video_path, project_path, skip_unsilence
        )
        # Checkpoint is saved within handle_audio_video on success
        current_stage = audio_processed_stage
    else:
        # Load paths from checkpoint data
        if checkpoint and "processed_video_path" in checkpoint["data"] and "processed_audio_path" in checkpoint["data"]:
             processed_video_path = checkpoint["data"]["processed_video_path"]
             processed_audio_path = checkpoint["data"]["processed_audio_path"]
             logger.info("Audio processing already completed. Using cached paths.")
        else:
             # Handle missing checkpoint data
             raise RuntimeError("Audio processing checkpoint missing necessary paths. Please clear checkpoint or re-run.")

    # Ensure paths are valid after loading or processing
    if not processed_video_path or not os.path.exists(processed_video_path):
         raise FileNotFoundError(f"Processed video path not found or invalid after audio stage: {processed_video_path}")
    if not processed_audio_path or not os.path.exists(processed_audio_path):
         raise FileNotFoundError(f"Processed audio path not found or invalid after audio stage: {processed_audio_path}")


    # --- Stage 3 & 4: Transcription & Scene Detection ---
    transcription_complete_stage = CHECKPOINTS["TRANSCRIPTION_COMPLETE"]
    scenes_detected_stage = CHECKPOINTS["SCENES_DETECTED"] # Includes NO_SCENES_DETECTED state
    transcript = None
    scene_boundaries = None

    # We need both transcript and scenes to proceed to visual analysis stages
    # Check if both stages are complete based on the latest checkpoint
    if current_stage >= scenes_detected_stage or current_stage == NO_SCENES_DETECTED:
        logger.info("Attempting to load transcript and scene boundaries from previous stages...")
        # Load transcript (transcript_path already defined at function start)
        if os.path.exists(transcript_path):
             try:
                 transcript = load_transcript(transcript_path)
             except Exception as e:
                 logger.warning(f"Failed to load transcript from file ({transcript_path}): {e}. Will attempt re-run.")
                 current_stage = audio_processed_stage # Force re-run
        else:
            logger.warning("Transcription checkpoint likely passed, but transcript file missing. Re-running.")
            current_stage = audio_processed_stage # Force re-run

        # Load scene boundaries (handle NO_SCENES state)
        if current_stage >= scenes_detected_stage: # Check again after potential reset
            scene_boundaries = checkpoint["data"].get("scene_boundaries")
        elif current_stage == NO_SCENES_DETECTED:
             scene_boundaries = [] # Empty list signifies no scenes
        else:
             # This case shouldn't be reached if current_stage was reset correctly
             logger.warning("Scene detection state inconsistent. Re-running detection.")
             current_stage = audio_processed_stage

        # Final check if loaded data is valid
        if transcript is None or scene_boundaries is None:
             logger.warning("Failed to load necessary transcript/scene data. Re-running detection/transcription.")
             current_stage = audio_processed_stage # Ensure re-run

    # Run transcription and scene detection if not loaded successfully
    if current_stage < scenes_detected_stage and current_stage != NO_SCENES_DETECTED:
        logger.info("Starting transcription and scene detection...")
        transcript, scene_boundaries = handle_transcription_and_scene_detection(
            audio_path=processed_audio_path,
            video_path=processed_video_path, # Use processed video for scene detection
            project_path=project_path,
            api=api,
            scene_threshold=scene_threshold,
            min_scene_len_sec=min_scene_len,
        )
        # Checkpoints are saved within handle_transcription_and_scene_detection
        if transcript is not None and scene_boundaries is not None:
             # Update stage based on whether scenes were found
             last_checkpoint = load_checkpoint(project_path) # Reload to get latest stage
             current_stage = last_checkpoint["stage"] if last_checkpoint else current_stage
        else:
             logger.error("Failed to get transcript or detect scenes. Cannot proceed.")
             return {
                 "error": "Transcription or Scene Detection failed.",
                 "transcript": transcript, # Return partial results
                 "scene_boundaries": scene_boundaries
             }

    # --- Stage 5: Scene Visual Analysis ---
    visual_analysis_complete_stage = CHECKPOINTS["VISUAL_ANALYSIS_COMPLETE"]
    analyzed_scenes_results = None
    scene_analysis_results_path = os.path.join(project_path, "scene_analysis", "scene_analysis_results.json") # Define path

    if current_stage < visual_analysis_complete_stage:
        logger.info("Starting scene visual analysis...")
        if not scene_boundaries:
            logger.info("Skipping visual analysis as no scenes were detected.")
            analyzed_scenes_results = [] # Need empty list for consistency
            # Save checkpoint indicating skipped analysis? Or rely on next stage check?
            save_checkpoint(
                 project_path, visual_analysis_complete_stage, {"analyzed_scenes_results_path": None, "skipped": True}
            )
            current_stage = visual_analysis_complete_stage
        else:
            try:
                # Initialize Gemini Client (ensure API key is available via env var)
                gemini_client = GeminiClient() # Assumes API key is in env
                analyzed_scenes_results = analyze_scenes(
                    input_video=processed_video_path, # Use the processed video
                    scene_boundaries=scene_boundaries,
                    project_path=project_path,
                    gemini_client=gemini_client,
                    software_list=software_list,
                    ocr_lang=ocr_lang,
                    frames_per_scene=frames_per_scene,
                    frame_format=frame_format,
                    compression_quality=compression_quality,
                    register=register, # Pass register
                    visual_similarity_threshold=visual_similarity_threshold # Pass threshold
                )
                # analyze_scenes saves its own results and checkpoint internally now
                # Reload checkpoint to confirm stage update
                last_checkpoint = load_checkpoint(project_path)
                current_stage = last_checkpoint["stage"] if last_checkpoint and last_checkpoint["stage"] == visual_analysis_complete_stage else current_stage

            except Exception as e:
                 logger.error(f"Scene visual analysis failed: {e}", exc_info=True)
                 return {"error": f"Visual analysis failed: {e}"}
    else:
        logger.info("Visual analysis already completed.")
        # Load results from file if stage was already complete
        if os.path.exists(scene_analysis_results_path):
             try:
                 from .analysis.visual_analysis import load_analyzed_scenes # Local import ok?
                 analyzed_scenes_results = load_analyzed_scenes(os.path.join(project_path, "scene_analysis"))
                 if analyzed_scenes_results is None: # Handle empty list case
                      analyzed_scenes_results = []
             except Exception as e:
                  logger.warning(f"Failed to load existing visual analysis results: {e}")
                  # Consider re-running by resetting stage? For now, proceed cautiously.
                  analyzed_scenes_results = []
        else:
             # If checkpoint says complete but file missing, log warning
             logger.warning(f"Visual analysis checkpoint complete, but results file missing: {scene_analysis_results_path}")
             # If no scenes were detected previously, ensure results list is empty
             if load_checkpoint(project_path).get("stage") == NO_SCENES_DETECTED:
                  analyzed_scenes_results = []
             else:
                 # This indicates a problem, maybe reset stage and force re-run?
                 logger.error("Inconsistent state: Visual analysis checkpoint passed but results missing.")
                 return {"error": "Inconsistent state: Missing visual analysis results."}


    # --- Stage 6: Visual Topic Modeling ---
    visual_topic_modeling_complete_stage = CHECKPOINTS["VISUAL_TOPIC_MODELING_COMPLETE"]
    visual_topic_results = None
    visual_topic_results_path = os.path.join(project_path, "visual_topic_analysis_results.json") # Define path

    if current_stage < visual_topic_modeling_complete_stage:
        logger.info("Starting visual topic modeling...")
        if not transcript:
             logger.error("Transcript not available, cannot perform visual topic modeling.")
             return {"error": "Transcript missing for visual topic modeling."}
        if analyzed_scenes_results is None:
             logger.error("Analyzed scenes results not available. Cannot perform visual topic modeling.")
             return {"error": "Missing analyzed scenes results."}

        try:
            # Prepare visual frames input
            visual_frames = prepare_visual_frames_for_topic_modeling(analyzed_scenes_results)

            # Run the combined analysis
            visual_topic_results = process_transcript_with_visuals(
                transcript_sentences=transcript, # Use the loaded transcript
                visual_frames=visual_frames,
                project_path=project_path,
                register=register
            )
            # process_transcript_with_visuals saves its own results and checkpoint
            last_checkpoint = load_checkpoint(project_path) # Reload checkpoint
            current_stage = last_checkpoint["stage"] if last_checkpoint and last_checkpoint["stage"] == visual_topic_modeling_complete_stage else current_stage

        except Exception as e:
             logger.error(f"Visual topic modeling failed: {e}", exc_info=True)
             return {"error": f"Visual topic modeling failed: {e}"}
    else:
        logger.info("Visual topic modeling already completed.")
        # Load existing results
        if os.path.exists(visual_topic_results_path):
             try:
                 with open(visual_topic_results_path, 'r', encoding='utf-8') as f:
                      visual_topic_results = json.load(f)
             except Exception as e:
                  logger.warning(f"Failed to load existing visual topic results: {e}")
                  visual_topic_results = None # Ensure it's None if loading fails
        else:
             logger.warning(f"Visual topic modeling checkpoint complete, but results file missing: {visual_topic_results_path}")
             visual_topic_results = None


    # --- Stage 7: Video Splitting ---
    video_split_complete_stage = CHECKPOINTS["VIDEO_SPLIT_COMPLETE"]
    split_video_paths = None

    if current_stage < video_split_complete_stage:
         logger.info("Splitting video based on identified topic segments...")
         if visual_topic_results is None:
             logger.warning("Visual topic modeling results not available. Cannot split video.")
             # Save checkpoint indicating skipped split?
         else:
              final_segments = visual_topic_results.get("segments", [])
              if not final_segments:
                   logger.warning("No final segments identified by visual topic modeling. Cannot split video.")
              else:
                   # Extract start/end times for split_video_by_scenes
                   segment_boundaries = [(seg.get("start_time", 0.0), seg.get("end_time", 0.0)) for seg in final_segments]
                   split_output_dir = os.path.join(project_path, "split_videos")
                   os.makedirs(split_output_dir, exist_ok=True)
                   try:
                        split_video_paths = split_video_by_scenes(
                             video_path=processed_video_path, # Use processed video
                             scene_list=segment_boundaries, # Use boundaries from topic analysis
                             output_dir=split_output_dir,
                        )
                        save_checkpoint(
                             project_path, video_split_complete_stage, {"split_video_paths": split_video_paths}
                        )
                        current_stage = video_split_complete_stage
                        logger.info(f"Video successfully split into {len(split_video_paths)} topic-based segments.")
                   except Exception as e:
                        logger.error(f"Failed to split video based on topic segments: {e}", exc_info=True)
                        split_video_paths = [] # Indicate failure
    else:
        logger.info("Video splitting already completed.")
        # Load split_video_paths from checkpoint data if needed
        if checkpoint and "split_video_paths" in checkpoint["data"]:
            split_video_paths = checkpoint["data"]["split_video_paths"]


    # --- Stage 8: Final Results ---
    process_complete_stage = CHECKPOINTS["PROCESS_COMPLETE"]
    logger.info("Preparing final results...")

    # Ensure key results are loaded or available
    if visual_topic_results is None and os.path.exists(visual_topic_results_path):
        logger.info("Reloading visual topic results for final summary...")
        try:
             with open(visual_topic_results_path, 'r', encoding='utf-8') as f:
                  visual_topic_results = json.load(f)
        except Exception:
             logger.error("Failed to reload visual topic results for final summary.")

    final_results = {
        "project_path": project_path,
        "original_input": input_path,
        "processed_video_path": processed_video_path,
        "processed_audio_path": processed_audio_path,
        # Include the main results from the visual topic modeling stage
        "visual_topic_analysis": visual_topic_results if visual_topic_results else {"error": "Results unavailable"},
        "split_video_paths": split_video_paths if split_video_paths is not None else [],
        # Add references to other intermediate files if useful
        "transcript_path": transcript_path if transcript else None,
        "scene_analysis_path": scene_analysis_results_path if analyzed_scenes_results is not None else None,
    }
    # Save final results JSON
    final_results_path = os.path.join(project_path, "final_results.json")
    logger.info(f"Saving final results to: {final_results_path}")
    try:
        with open(final_results_path, "w", encoding="utf-8") as f:
            # Use default=str for safety with potential non-serializable types
            json.dump(final_results, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        logger.error(f"Failed to save final results JSON: {e}")

    # Final overall completion checkpoint
    save_checkpoint(project_path, process_complete_stage, {"final_results_path": final_results_path})
    logger.info("Unified processing pipeline complete.")

    return final_results
