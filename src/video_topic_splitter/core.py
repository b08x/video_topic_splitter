#!/usr/bin/env python3
"""Core processing functionality for video scene splitter."""

import json
import logging
import os
import shutil

from deepgram import DeepgramClient, PrerecordedOptions
from dotenv import load_dotenv
# Removed Groq import as it's not the primary focus now
# from groq import Groq

# Removed topic modeling import
# from .analysis.topic_modeling import process_transcript
from .analysis.visual_analysis import analyze_scenes, save_analyzed_scenes # Use analyze_scenes
from .api.deepgram import transcribe_file_deepgram
from .constants import CHECKPOINTS
from .processing.audio.audio import (convert_to_mono_and_resample,
                                     extract_audio, normalize_audio,
                                     remove_silence)
# Import scene detection and splitting functions
from .processing.video.scene_detection import (detect_scenes,
                                               split_video_by_scenes)
from .project import load_checkpoint, save_checkpoint
from .transcription import load_transcript, save_transcript, save_transcription
from .utils.youtube import download_video, is_youtube_url # Added is_youtube_url

load_dotenv()
logger = logging.getLogger(__name__)

# Keep handle_audio_video largely the same, as audio processing is still needed
def handle_audio_video(video_path, project_path, skip_unsilence=False):
    """
    Processes the audio track of a video file: normalization, optional silence
    removal, extraction, and resampling, with checkpointing.

    Args:
        video_path (str): Path to the input video file.
        project_path (str): Path to the project directory.
        skip_unsilence (bool, optional): If True, skips silence removal. Defaults to False.

    Returns:
        tuple[str, str]: Paths to the (potentially unsilenced) video and the
                         final mono, resampled audio file.

    Raises:
        RuntimeError: If any audio processing step fails critically.
        FileNotFoundError: If input video_path does not exist.
    """
    audio_dir = os.path.join(project_path, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    logger.info(f"Audio processing outputs will be saved in: {audio_dir}")

    video_name, video_ext = os.path.splitext(os.path.basename(video_path))
    normalized_video_path = os.path.join(project_path, f"normalized_video{video_ext}")
    unsilenced_video_path = os.path.join(project_path, f"unsilenced_video{video_ext}")
    raw_audio_path = os.path.join(audio_dir, "extracted_audio.opus")
    mono_resampled_audio_path = os.path.join(audio_dir, "mono_resampled_audio.m4a")

    # Check for existing final processed files first
    if os.path.exists(unsilenced_video_path) and os.path.exists(mono_resampled_audio_path):
        logger.info("Found existing processed audio/video files. Using cached versions.")
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
                "processed_video_path": unsilenced_video_path, # Renamed key for clarity
                "processed_audio_path": mono_resampled_audio_path, # Renamed key
            },
        )
        logger.info("Audio processing checkpoint saved.")
    else:
        logger.error("Final processed audio/video files not found. Checkpoint not saved.")
        raise RuntimeError("Audio processing finished but expected output files are missing.")

    return unsilenced_video_path, mono_resampled_audio_path


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

    Args:
        audio_path (str): Path to the processed audio file for transcription.
        video_path (str): Path to the video file for scene detection.
        project_path (str): Path to the project directory.
        api (str, optional): Transcription API ('deepgram'). Defaults to "deepgram".
        scene_threshold (float, optional): Threshold for scene detection. Defaults to 27.0.
        min_scene_len_sec (float, optional): Min scene length in seconds. Defaults to 1.0.

    Returns:
        tuple[Optional[list], Optional[list]]: A tuple containing:
            - transcript (list | None): List of transcript utterances or None on failure.
            - scene_boundaries (list | None): List of (start_sec, end_sec) tuples for scenes or None.
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
                # Use options suitable for getting timed utterances
                deepgram_options = PrerecordedOptions(
                    model="nova-2", language="en", smart_format=True,
                    punctuate=True, utterances=True,
                    # Add paragraphs=True, diarize=True if needed for context later
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
            raise ValueError(f"Transcription API '{api}' is not currently supported.")

    # Checkpoint after transcription attempt
    if transcript is not None:
        save_checkpoint(
            project_path, CHECKPOINTS["TRANSCRIPTION_COMPLETE"], {"transcript_path": transcript_path}
        )
        logger.info("Transcription checkpoint saved.")
    else:
        logger.error("Transcription failed or produced no utterances.")
        # Decide whether to proceed without transcript or raise error
        # For now, return None for transcript

    # --- Scene Detection ---
    scene_boundaries = None
    scenes_csv_path = os.path.join(project_path, "scenes", "scenes.csv") # Standard path for CSV
    if os.path.exists(scenes_csv_path):
        logger.info("Loading existing scene boundaries from CSV...")
        # Basic loading logic, assumes CSV format from detect_scenes
        try:
             scene_boundaries = []
             with open(scenes_csv_path, 'r', encoding='utf-8') as f:
                 reader = csv.reader(f)
                 header = next(reader) # Skip header
                 for row in reader:
                     try:
                         # Assuming start_sec is col 3 (index 3) and end_sec is col 4 (index 4)
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
        try:
            scene_boundaries = detect_scenes(
                video_path=video_path,
                output_dir=scenes_output_dir,
                threshold=scene_threshold,
                min_scene_len_sec=min_scene_len_sec,
                save_csv=True # Ensure CSV is saved
            )
            logger.info(f"Scene detection complete. Found {len(scene_boundaries)} scenes.")
        except Exception as e:
            logger.error(f"Scene detection failed: {e}", exc_info=True)
            scene_boundaries = None # Indicate failure

    # Checkpoint after scene detection attempt
    if scene_boundaries is not None:
        save_checkpoint(
            project_path, CHECKPOINTS["SCENES_DETECTED"], {"scene_boundaries": scene_boundaries}
        )
        logger.info("Scene detection checkpoint saved.")
    else:
        logger.error("Scene detection failed.")
        # Decide whether to proceed or raise error
        # For now, return None for scenes

    return transcript, scene_boundaries


def process_video(
    input_path: str, # Can be local path or YouTube URL
    project_path: str,
    api: str = "deepgram",
    skip_unsilence: bool = False,
    # Removed transcribe_only flag for simplification
    # Removed topic modeling params (num_topics)
    # --- Scene Detection Params ---
    scene_threshold: float = 27.0,
    min_scene_len: float = 1.0,
    # --- Visual Analysis Params ---
    software_list: Optional[list] = None,
    ocr_lang: str = "eng",
    frames_per_scene: int = 1,
    frame_format: str = "jpg",
    compression_quality: int = 90,
    register: str = "it-workflow", # Keep register for Gemini context
    # Removed logo params
    # Removed thumbnail params (handled within scene analysis if needed)
):
    """
    Main pipeline function to process a video: download (optional), process audio,
    transcribe, detect scenes, analyze scenes visually, and split video by scenes.

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
        register (str, optional): Analysis register for Gemini. Defaults to "it-workflow".

    Returns:
        dict: Final results including transcript, scene analysis, and split video paths.

    Raises:
        RuntimeError, ValueError, FileNotFoundError, Exception: Propagated from sub-functions.
    """
    os.makedirs(project_path, exist_ok=True)
    logger.info(f"Starting processing for input: {input_path}")
    logger.info(f"Project directory: {project_path}")

    checkpoint = load_checkpoint(project_path)
    current_stage = checkpoint["stage"] if checkpoint else -1
    logger.info(f"Current checkpoint stage: {current_stage}")

    is_youtube = is_youtube_url(input_path)
    video_path = input_path # May be updated after download

    # --- Stage 1: YouTube Download (if applicable) ---
    youtube_complete_stage = CHECKPOINTS["YOUTUBE_DOWNLOAD_COMPLETE"]
    if is_youtube and current_stage < youtube_complete_stage:
        logger.info("Downloading YouTube video...")
        download_path = os.path.join(project_path, "source_video.mp4")
        result = download_video(input_path, download_path, project_path)
        if result["status"] == "error":
            raise RuntimeError(f"YouTube download failed: {result['message']}")
        video_path = result["file_path"] # Use the actual downloaded path
        save_checkpoint(project_path, youtube_complete_stage, {"video_path": video_path})
        logger.info(f"YouTube video downloaded to: {video_path}")
        current_stage = youtube_complete_stage
    elif is_youtube:
        video_path = checkpoint["data"]["video_path"]
        logger.info(f"Using previously downloaded YouTube video: {video_path}")
    elif not os.path.exists(video_path):
         # If it's not YouTube and doesn't exist locally
         raise FileNotFoundError(f"Input video file not found: {video_path}")


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
        processed_video_path = checkpoint["data"]["processed_video_path"]
        processed_audio_path = checkpoint["data"]["processed_audio_path"]
        logger.info("Audio processing already completed. Using cached paths.")

    # Ensure paths are valid after audio processing step
    if not processed_video_path or not os.path.exists(processed_video_path):
         raise FileNotFoundError(f"Processed video path not found or invalid after audio stage: {processed_video_path}")
    if not processed_audio_path or not os.path.exists(processed_audio_path):
         raise FileNotFoundError(f"Processed audio path not found or invalid after audio stage: {processed_audio_path}")


    # --- Stage 3 & 4: Transcription & Scene Detection ---
    transcription_complete_stage = CHECKPOINTS["TRANSCRIPTION_COMPLETE"]
    scenes_detected_stage = CHECKPOINTS["SCENES_DETECTED"]
    transcript = None
    scene_boundaries = None

    # Try to load from SCENES_DETECTED checkpoint first
    if current_stage >= scenes_detected_stage:
         logger.info("Loading transcript and scene boundaries from checkpoint...")
         # Need to ensure transcript path was also saved or load it separately
         transcript_path = os.path.join(project_path, "transcript.json")
         if os.path.exists(transcript_path):
             try:
                 transcript = load_transcript(transcript_path)
             except Exception as e:
                 logger.warning(f"Failed to load transcript from file ({transcript_path}) even though checkpoint exists: {e}")
                 # Force re-transcription/detection below
                 current_stage = audio_processed_stage # Reset stage
         else:
              logger.warning("Scenes detected checkpoint exists, but transcript file is missing. Re-running.")
              current_stage = audio_processed_stage # Reset stage

         if current_stage >= scenes_detected_stage: # Re-check stage
             scene_boundaries = checkpoint["data"].get("scene_boundaries")
             if transcript is None or scene_boundaries is None:
                  logger.warning("Checkpoint data incomplete for transcript/scenes. Re-running detection.")
                  current_stage = audio_processed_stage # Reset stage
             else:
                  logger.info("Transcript and scene boundaries loaded from checkpoint/files.")

    # If not loaded from checkpoint, run transcription and scene detection
    if current_stage < scenes_detected_stage:
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
             current_stage = scenes_detected_stage # Update stage only if both succeed
        else:
             logger.error("Failed to get transcript or detect scenes. Cannot proceed.")
             # Return partial results or raise error
             return {
                 "error": "Transcription or Scene Detection failed.",
                 "transcript": transcript,
                 "scene_boundaries": scene_boundaries
             }

    # --- Stage 5: Scene Visual Analysis ---
    scene_analysis_complete_stage = CHECKPOINTS["SCENE_ANALYSIS_COMPLETE"]
    analyzed_scenes = None
    if current_stage < scene_analysis_complete_stage:
        logger.info("Starting scene visual analysis...")
        if not scene_boundaries:
             raise RuntimeError("Cannot perform scene analysis without scene boundaries.")

        analyzed_scenes = analyze_scenes(
            input_video=processed_video_path, # Analyze the processed video
            scene_boundaries=scene_boundaries,
            project_path=project_path,
            software_list=software_list,
            ocr_lang=ocr_lang,
            frames_per_scene=frames_per_scene,
            frame_format=frame_format,
            compression_quality=compression_quality,
            register=register,
        )
        # Checkpoint is saved within analyze_scenes on success
        current_stage = scene_analysis_complete_stage
    else:
        logger.info("Scene analysis already completed. Loading results...")
        # Load results from checkpoint or file
        analysis_results_path = os.path.join(project_path, "scene_analysis", "scene_analysis_results.json")
        if os.path.exists(analysis_results_path):
             try:
                 with open(analysis_results_path, 'r', encoding='utf-8') as f:
                     analyzed_scenes = json.load(f)
                 logger.info("Loaded scene analysis results from file.")
             except Exception as e:
                 logger.error(f"Failed to load scene analysis results from {analysis_results_path}: {e}")
                 # Decide how to handle - raise error or return incomplete?
                 raise RuntimeError("Failed to load existing scene analysis results.") from e
        else:
             logger.error("Scene analysis checkpoint indicates completion, but results file is missing.")
             raise RuntimeError("Scene analysis results file missing despite checkpoint.")


    # --- Stage 6: Video Splitting ---
    video_split_complete_stage = CHECKPOINTS["VIDEO_SPLIT_COMPLETE"]
    split_video_paths = None
    if current_stage < video_split_complete_stage:
        logger.info("Splitting video by detected scenes...")
        if not scene_boundaries:
             raise RuntimeError("Cannot split video without scene boundaries.")

        split_output_dir = os.path.join(project_path, "split_videos")
        os.makedirs(split_output_dir, exist_ok=True)
        try:
            split_video_paths = split_video_by_scenes(
                video_path=processed_video_path, # Split the same video used for analysis
                scene_list=scene_boundaries,
                output_dir=split_output_dir,
                # output_file_template defaults to 'scene_$SCENE_NUMBER.mp4'
            )
            save_checkpoint(
                project_path, video_split_complete_stage, {"split_video_paths": split_video_paths}
            )
            logger.info(f"Video successfully split into {len(split_video_paths)} segments.")
            current_stage = video_split_complete_stage
        except Exception as e:
             logger.error(f"Failed to split video: {e}", exc_info=True)
             # Don't raise error, just report failure and proceed without split paths
             split_video_paths = [] # Indicate failure
    else:
        logger.info("Video splitting already completed.")
        split_video_paths = checkpoint["data"].get("split_video_paths")
        if split_video_paths is None:
             logger.warning("Video split checkpoint exists, but path list is missing. Cannot confirm split files.")


    # --- Final Results ---
    final_results = {
        "project_path": project_path,
        "original_input": input_path,
        "processed_video_path": processed_video_path,
        "processed_audio_path": processed_audio_path,
        "transcript": transcript, # Include the simplified transcript
        "scene_boundaries": scene_boundaries,
        "scene_analysis": analyzed_scenes,
        "split_video_paths": split_video_paths,
    }

    # Save final results JSON
    final_results_path = os.path.join(project_path, "final_results.json")
    logger.info(f"Saving final results to: {final_results_path}")
    try:
        with open(final_results_path, "w", encoding="utf-8") as f:
            # Use default=str to handle potential non-serializable types like Timecode
            json.dump(final_results, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        logger.error(f"Failed to save final results JSON: {e}")

    # Final overall completion checkpoint
    save_checkpoint(project_path, CHECKPOINTS["PROCESS_COMPLETE"], {"final_results_path": final_results_path})
    logger.info("Processing complete. Final checkpoint saved.")

    return final_results

