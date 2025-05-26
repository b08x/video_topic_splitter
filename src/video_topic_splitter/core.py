# core/core.py
#!/usr/bin/env python3
"""
Core logic for video segmentation and analysis.
Orchestrates the processing pipeline.
"""

import re
import logging
import os
import time
from typing import List, Tuple, Optional, Dict, Any
import json
import glob
from moviepy.editor import VideoFileClip

# Import necessary modules
try:
    from .api.gemini import GeminiClient
    # --- Add import for reading CSV ---
    from .processing.video import scene_detection, video_segmentation
    # --- End Add ---
    from .processing.transcript import transcript_processing
except ImportError:
    # Fallback for different execution contexts if needed
    from api.gemini import GeminiClient
    # --- Add import for reading CSV ---
    from processing.video import scene_detection, video_segmentation
    # --- End Add ---
    from processing.transcript import transcript_processing

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Orchestrates the video processing pipeline.
    """

    def __init__(self, gemini_client: GeminiClient):
        """
        Initializes the VideoProcessor with a GeminiClient instance.

        Args:
            gemini_client: An instance of the GeminiClient for API interactions.
        """
        self.gemini_client = gemini_client

    def _write_analysis_to_json(self, data: Dict[str, Any], output_dir: str, filename: str = "analysis_results.json"):
        """
        Writes the analysis data (including metadata and segments) to a JSON file.

        Args:
            data: The dictionary containing analysis results (metadata, segments, errors).
            output_dir: The directory to save the JSON file.
            filename: The name for the output JSON file.
        """
        output_path = os.path.join(output_dir, filename)
        try:
            os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            logger.info(
                f"Successfully wrote analysis results to {output_path}")
        except IOError as e:
            logger.error(
                f"Failed to write analysis results to {output_path}: {e}", exc_info=True)
        except TypeError as e:
            logger.error(
                f"Failed to serialize analysis data to JSON: {e}", exc_info=True)
            # Optionally write a partial or error state file
            try:
                # Attempt to write at least the error info
                error_data = data.copy()  # Avoid modifying original data if possible
                error_data["serialization_error"] = f"Failed to serialize data: {e}"
                with open(output_path, "w", encoding="utf-8") as f:
                    # Try dumping what we can, might still fail if complex objects are the issue
                    # Use default=str as fallback
                    json.dump(error_data, f, indent=4, default=str)
                logger.warning(
                    f"Wrote partial/error information to {output_path}")
            except Exception as write_err:
                logger.error(
                    f"Could not even write error file to {output_path}: {write_err}")

    def process_video(
        self, video_path: str, output_dir: str, use_scene_detection: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Processes the video, segmenting it, analyzing the segments, and saving the results to JSON.

        Args:
            video_path: Path to the input video file.
            output_dir: Directory to store output files (e.g., segments, analysis results).
            use_scene_detection: Whether to use scene detection as a first pass.

        Returns:
            A dictionary containing the processing metadata and segment analysis results,
            or None if a critical error occurred early on.
        """
        logger.info(f"Processing video: {video_path}")
        segments_data: List[dict] = []
        processing_metadata: Dict[str, Any] = {
            "video_path": video_path,
            "output_dir": output_dir,
            "use_scene_detection_flag": use_scene_detection,
            "parsed_transcript_sentences": 0,
            "parsed_gemini_scene_boundaries": 0,
            "pyscenedetect_boundaries_found": 0,
            "pyscenedetect_run": False,  # Initialize here
            "pyscenedetect_succeeded": False,  # Initialize here
            # 'Gemini', 'PySceneDetect', 'PySceneDetect (Existing CSV)', 'Fallback', 'None'
            "final_boundaries_source": "None",
        }
        errors_list: List[Dict[str, str]] = []
        output_json_filename = "analysis_results.json"

        final_results: Dict[str, Any] = {
            "processing_metadata": processing_metadata,
            "segments": segments_data,
            "errors": errors_list
        }

        try:
            # Ensure the main output directory exists early
            os.makedirs(output_dir, exist_ok=True)

            # 1. Generate transcript and optionally detect scene changes using Gemini
            # ... (Gemini transcript/boundary fetching remains the same) ...
            gemini_response_text = self._generate_transcript_and_metadata(
                video_path)

            if "Analysis failed" in gemini_response_text:
                error_msg = f"Gemini analysis failed early: {gemini_response_text}"
                logger.error(error_msg)
                errors_list.append(
                    {"step": "gemini_metadata", "message": error_msg})
                self._write_analysis_to_json(
                    final_results, output_dir, output_json_filename)
                return None

            transcript = self._parse_transcript(gemini_response_text)
            processing_metadata["parsed_transcript_sentences"] = len(
                transcript)

            gemini_scene_boundaries = self._parse_scene_boundaries(
                gemini_response_text)
            processing_metadata["parsed_gemini_scene_boundaries"] = len(
                gemini_scene_boundaries)

            # 2. Detect scene changes using PySceneDetect or load existing results
            scene_boundaries: List[Tuple[float, float]] = []
            # Metadata flags moved to initialization

            if use_scene_detection:
                # Mark that we intended to use it
                processing_metadata["pyscenedetect_run"] = True
                logger.info("Checking for PySceneDetect results...")
                scene_detection_dir = os.path.join(
                    output_dir, "scene_detection")
                os.makedirs(scene_detection_dir, exist_ok=True)

                # Construct the expected CSV path
                # scene_csv_path = os.path.join(
                #     scene_detection_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}-Scenes.csv")
                scene_csv_path = os.path.join(scene_detection_dir, "scenes.csv") # Alternative fixed name

                # --- Modified Logic: Check and Load or Run ---
                if os.path.exists(scene_csv_path):
                    logger.info(
                        f"Found existing scene file: '{scene_csv_path}'. Attempting to load boundaries from it.")
                    try:
                        # *** Assumes scene_detection.read_scenes_from_csv exists ***
                        # This function should read the CSV and return List[Tuple[float, float]]
                        # It should handle potential errors during file reading/parsing.
                        scene_boundaries = scene_detection.read_scenes_from_csv(
                            scene_csv_path)
                        processing_metadata["pyscenedetect_boundaries_found"] = len(
                            scene_boundaries)
                        processing_metadata[
                            "final_boundaries_source"] = "PySceneDetect (Existing CSV)"
                        # We successfully got boundaries
                        processing_metadata["pyscenedetect_succeeded"] = True
                        logger.info(
                            f"Successfully loaded {len(scene_boundaries)} scene boundaries from existing CSV.")
                    except Exception as csv_err:
                        error_msg = f"Failed to read or parse existing scene CSV '{scene_csv_path}': {csv_err}. Will attempt to run detection."
                        logger.error(error_msg, exc_info=True)
                        errors_list.append(
                            {"step": "pyscenedetect_read_csv", "message": error_msg})
                        # Reset flags as we didn't succeed in loading
                        processing_metadata["pyscenedetect_succeeded"] = False
                        processing_metadata["final_boundaries_source"] = "None (CSV Read Error)"
                        # Fall through to run detection below
                        scene_boundaries = []  # Ensure it's empty before trying detection

                # If CSV didn't exist OR reading it failed, run detection
                if not processing_metadata["pyscenedetect_succeeded"]:
                    logger.info(
                        "Existing scene CSV not found or failed to load. Running PySceneDetect.")
                    try:
                        detected_scenes = scene_detection.detect_scenes(
                            video_path, scene_detection_dir)
                        scene_boundaries = detected_scenes
                        processing_metadata["pyscenedetect_boundaries_found"] = len(
                            detected_scenes)
                        processing_metadata["final_boundaries_source"] = "PySceneDetect"
                        processing_metadata["pyscenedetect_succeeded"] = True
                        logger.info(
                            f"PySceneDetect successfully found {len(detected_scenes)} scenes.")
                    except Exception as sd_err:
                        error_msg = f"PySceneDetect failed: {sd_err}"
                        logger.error(error_msg, exc_info=True)
                        errors_list.append(
                            {"step": "pyscenedetect_run", "message": error_msg})
                        # Mark failure
                        processing_metadata["pyscenedetect_succeeded"] = False
                        # Fallback logic (only if detection failed)
                        if gemini_scene_boundaries:
                            logger.warning(
                                "Falling back to Gemini scene boundaries after PySceneDetect error.")
                            scene_boundaries = gemini_scene_boundaries
                            processing_metadata["final_boundaries_source"] = "Fallback (Gemini)"
                        else:
                            logger.warning(
                                "No scene boundaries available after PySceneDetect error.")
                            processing_metadata["final_boundaries_source"] = "None (PySceneDetect Error)"
                            scene_boundaries = []  # Ensure empty list

            # --- End Modified Logic ---

            # If scene detection was disabled or failed without fallback, check Gemini
            elif gemini_scene_boundaries:
                logger.info(
                    "Using scene boundaries provided by Gemini (PySceneDetect disabled or failed without fallback).")
                scene_boundaries = gemini_scene_boundaries
                processing_metadata["final_boundaries_source"] = "Gemini"
            else:
                logger.info(
                    "Scene detection disabled and no Gemini boundaries found.")
                processing_metadata["final_boundaries_source"] = "None (Disabled/Not Found)"

            # 3. Segment and analyze each scene
            # ... (rest of the segmentation/analysis loop remains the same) ...
            if scene_boundaries:
                logger.info(
                    f"Processing {len(scene_boundaries)} detected scenes using boundaries from '{processing_metadata['final_boundaries_source']}'.")
                for i, (start_time, end_time) in enumerate(scene_boundaries):
                    logger.info(
                        f"Processing Scene {i+1}: {start_time:.2f}s - {end_time:.2f}s")
                    scene_output_dir = os.path.join(
                        output_dir, f"scene_{i+1:03d}")
                    os.makedirs(scene_output_dir, exist_ok=True)
                    try:
                        scene_segments = self._process_scene(
                            video_path, scene_output_dir, transcript, start_time, end_time
                        )
                        segments_data.extend(scene_segments)
                    except Exception as scene_err:
                        error_msg = f"Error processing scene {i+1} ({start_time:.2f}s-{end_time:.2f}s): {scene_err}"
                        logger.error(error_msg, exc_info=True)
                        errors_list.append(
                            {"step": f"process_scene_{i+1}", "message": error_msg})

            else:
                # Process the entire video as one segment
                logger.info(
                    "No scene boundaries defined, processing entire video as one scene.")
                try:
                    # Use the main output_dir for segments when processing as one scene
                    all_segments = self._process_scene(
                        video_path, output_dir, transcript)
                    segments_data.extend(all_segments)
                except Exception as full_video_err:
                    error_msg = f"Error processing full video as one scene: {full_video_err}"
                    logger.error(error_msg, exc_info=True)
                    errors_list.append(
                        {"step": "process_full_video", "message": error_msg})

            logger.info(
                f"Finished processing. Generated data for {len(segments_data)} segments.")

            # 4. Write the final results to JSON
            final_results["segments"] = segments_data
            final_results["errors"] = errors_list
            self._write_analysis_to_json(
                final_results, output_dir, output_json_filename)

            return final_results

        except Exception as e:
            error_msg = f"Critical error during video processing '{video_path}': {e}"
            logger.error(error_msg, exc_info=True)
            errors_list.append(
                {"step": "main_process_video", "message": error_msg})
            final_results["segments"] = segments_data
            final_results["errors"] = errors_list
            self._write_analysis_to_json(
                final_results, output_dir, output_json_filename)
            return None

    def _generate_transcript_and_metadata(self, video_path: str) -> str:
        """
        Generates the transcript and optionally detects scene changes using the Gemini API.

        Args:
            video_path: Path to the input video file.

        Returns:
            The raw string response from the Gemini API.
        """
        gemini_prompt = (
            "Transcribe this video, providing timestamps for each utterance in a JSON list format. "
            "Each item should have 'start_time', 'end_time', and 'transcript' keys. "
            "If possible, also identify scene changes and include them in the JSON response "
            "as a list under the key 'scene_boundaries', where each item has 'start_time' and 'end_time'."
            " Respond ONLY with the JSON object."  # Explicitly ask for JSON only
        )
        logger.info("Requesting transcript and scene boundaries from Gemini...")
        gemini_response = self.gemini_client.analyze(gemini_prompt, video_path)
        # Log beginning of response
        logger.debug(
            f"Raw Gemini response for metadata: {gemini_response[:500]}...")
        return gemini_response

    def _process_scene(
        self,
        video_path: str,
        output_dir: str,  # Directory for this scene's segments
        transcript: List[dict],
        scene_start: Optional[float] = None,
        scene_end: Optional[float] = None,
    ) -> List[dict]:
        """
        Segments and analyzes a single scene (or the entire video). Checks for
        existing segment files and analysis files before starting segmentation.

        Args:
            video_path: Path to the input video file.
            output_dir: Directory to store output files for this scene/video.
            transcript: The transcript of the video.
            scene_start: Start time of the scene (optional, for scene-based processing).
            scene_end: End time of the scene (optional, for scene-based processing).

        Returns:
            A list of dictionaries, where each dictionary represents a video segment
            and its analysis.
        """
        scene_segments_data: List[dict] = []
        # Filter transcript for the current scene
        if scene_start is not None and scene_end is not None:
            scene_transcript = transcript_processing.filter_transcript_by_time_range(
                transcript, scene_start, scene_end
            )
            logger.info(
                f"Filtered transcript contains {len(scene_transcript)} items for the current scene.")
        else:
            scene_transcript = transcript  # Use the whole transcript
            logger.info(
                "Using full transcript as no scene boundaries were specified.")

        # --- Check for existing segment files before proceeding ---
        existing_segments = []
        try:
            # Use a pattern that matches the expected segment filenames
            segment_pattern = os.path.join(output_dir, "segment_*.mp4")
            existing_segments = glob.glob(segment_pattern)
            if existing_segments:
                logger.info(f"Found {len(existing_segments)} existing segment file(s) matching pattern '{segment_pattern}' in {output_dir}.")
        except Exception as e:
            logger.error(
                f"Error checking for existing segment files in {output_dir}: {e}", exc_info=True)
        # --- End Check ---

        # If we have existing segments, use them instead of creating new ones
        segment_timestamps_paths = []
        if existing_segments:
            logger.info("Using existing segment files instead of creating new ones.")
            for segment_path in existing_segments:
                try:
                    # Extract start and end times from the segment file if possible
                    # This is a fallback approach since we don't have the original timestamps
                    with VideoFileClip(segment_path) as clip:
                        # For existing segments where we don't know the exact timestamps,
                        # we'll use relative positions within the scene
                        if scene_start is not None and scene_end is not None:
                            # If we know scene boundaries, estimate segment position within scene
                            segment_duration = clip.duration
                            segment_index = int(os.path.basename(segment_path).split('_')[1])
                            total_segments = len(existing_segments)
                            scene_duration = scene_end - scene_start
                            
                            # Estimate start and end times based on segment position in scene
                            estimated_start = scene_start + (scene_duration * (segment_index - 1) / total_segments)
                            estimated_end = estimated_start + segment_duration
                            
                            # Ensure we don't exceed scene boundaries
                            estimated_start = max(estimated_start, scene_start)
                            estimated_end = min(estimated_end, scene_end)
                            
                            segment_timestamps_paths.append((estimated_start, estimated_end, segment_path))
                        else:
                            # If we don't know scene boundaries, just use 0 as start time
                            segment_timestamps_paths.append((0, clip.duration, segment_path))
                except Exception as e:
                    logger.error(f"Error processing existing segment {segment_path}: {e}", exc_info=True)
                    # Add with unknown timestamps
                    segment_timestamps_paths.append((0, 0, segment_path))
        else:
            # No existing segments found, proceed with normal segmentation
            logger.info("No existing segments found. Proceeding with segmentation.")
            
            # 1. Identify silent periods
            silent_periods = self._detect_silent_periods(scene_transcript)

            # 2. Segment the scene based on silent periods
            # Pass scene_start/end to determine segment boundaries correctly
            segment_timestamps_paths = self._align_timestamps_and_segment(
                video_path, output_dir, scene_start, scene_end, silent_periods
            )

        # 3. Analyze each segment
        if not segment_timestamps_paths:
            logger.warning("No segments were created or found for this scene.")
            return []

        logger.info(
            f"Analyzing {len(segment_timestamps_paths)} segments for this scene...")
        for segment_start, segment_end, segment_path in segment_timestamps_paths:
            if segment_path == "ERROR_PATH_MISSING":
                logger.error(
                    f"Skipping analysis for segment {segment_start}-{segment_end} due to missing file.")
                continue
            # Check if the segment file actually exists before analysis (belt-and-suspenders)
            if not os.path.exists(segment_path):
                logger.error(
                    f"Segment file reported by segmentation step not found: {segment_path}. Skipping analysis.")
                continue

            # Check if an analysis file already exists for this segment
            segment_dir = os.path.dirname(segment_path)
            segment_basename = os.path.basename(segment_path)
            segment_name = os.path.splitext(segment_basename)[0]  # Remove extension
            analysis_filename = f"{segment_name}_analysis.json"
            analysis_path = os.path.join(segment_dir, analysis_filename)
            
            analysis = None
            if os.path.exists(analysis_path):
                # Load existing analysis if available
                try:
                    logger.info(f"Found existing analysis file: {analysis_path}")
                    with open(analysis_path, 'r', encoding='utf-8') as f:
                        analysis = json.load(f)
                    # Remove metadata fields that should not be part of the analysis dict
                    if "segment_path" in analysis:
                        analysis.pop("segment_path")
                    if "analysis_timestamp" in analysis:
                        analysis.pop("analysis_timestamp")
                    logger.info(f"Successfully loaded existing analysis for {segment_path}")
                except Exception as e:
                    logger.error(f"Error loading existing analysis file {analysis_path}: {e}", exc_info=True)
                    analysis = None
            
            # If no valid existing analysis was loaded, perform analysis
            if analysis is None:
                logger.info(f"No valid existing analysis found for {segment_path}. Performing analysis.")
                analysis = self._analyze_segment(segment_path)
                # Note: _analyze_segment now saves the analysis to a JSON file

            scene_segments_data.append(
                {
                    "segment_start": segment_start,
                    "segment_end": segment_end,
                    "segment_path": segment_path,
                    "analysis": analysis,  # Contains parsed summary and topics
                }
            )

        return scene_segments_data

    # --- Parsing Methods ---

    def _parse_transcript(self, gemini_response: str) -> List[dict]:
        """
        Parses the transcript from the Gemini API response.
        Includes cleaning for potential markdown fences.

        Args:
            gemini_response: The raw text response from the Gemini API (JSON string).

        Returns:
            A list of transcript dictionaries.
        """
        if not gemini_response or not gemini_response.strip():
            logger.warning("Received empty Gemini response for transcript.")
            return []

        cleaned_text = gemini_response.strip()
        # Clean potential markdown fences
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:-3].strip()
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:-3].strip()

        if not cleaned_text:
            logger.warning(
                "Gemini response for transcript became empty after cleaning.")
            return []

        logger.debug(
            f"Attempting to parse transcript JSON: {cleaned_text[:100]}...")
        try:
            data = json.loads(cleaned_text)
            # Log type
            logger.debug(f"Successfully parsed JSON. Data type: {type(data)}")

            # --- Add detailed logging for dictionaries ---
            if isinstance(data, dict):
                logger.debug(
                    f"Parsed JSON dictionary keys: {list(data.keys())}")
                # Optional: Log the full dict if it's not excessively large for debugging
                # logger.debug(f"Parsed JSON dictionary content: {data}")
            # --- End Add ---

            # Check if the response is a dictionary or list
            if isinstance(data, dict):
                logger.debug(
                    "Parsed transcript response as dict. Looking for 'transcript', 'text', or 'utterances' key.")
                # Check for 'transcript' first, then 'text', then 'utterances'
                transcript = None
                if "transcript" in data and isinstance(data["transcript"], list):
                    transcript = data["transcript"]
                    logger.debug("Found 'transcript' key with a list.")
                elif "text" in data and isinstance(data["text"], list):
                    transcript = data["text"]
                    logger.debug("Found 'text' key with a list.")
                # --- Add check for 'utterances' ---
                elif "utterances" in data and isinstance(data["utterances"], list):
                    transcript = data["utterances"]
                    logger.debug("Found 'utterances' key with a list.")
                # --- End Add ---
                # You can add more elif checks here if Gemini uses other keys in the future

                if transcript is None:
                    # Log the structure that *was* received when the expected keys are missing
                    logger.warning(
                        f"Could not find a suitable transcript list within the dictionary response. Keys found: {list(data.keys())}")
                    # Log the problematic data structure for debugging
                    # Log the actual data
                    logger.debug(f"Problematic dictionary structure: {data}")
                    return []

                # --- Key Name Standardization (Important!) ---
                # Now that we've found the list (under 'transcript', 'text', or 'utterances'),
                # we need to make sure the *items* within that list have the keys our code expects later
                # (e.g., 'text', 'start_time', 'end_time').
                # The prompt asked for 'transcript' inside each item, but Gemini might return 'text' or something else.
                # Let's standardize the text key within each sentence dictionary.

                standardized_transcript = []
                for sentence_dict in transcript:
                    if not isinstance(sentence_dict, dict):
                        logger.warning(
                            f"Skipping non-dictionary item in transcript list: {sentence_dict}")
                        continue

                    # Find the actual text key ('transcript', 'text', 'utterance', etc.)
                    text_content = None
                    # Add more possibilities if needed
                    possible_text_keys = ["transcript",
                                          "text", "utterance", "content"]
                    found_key = None
                    for key in possible_text_keys:
                        if key in sentence_dict:
                            text_content = sentence_dict[key]
                            found_key = key
                            break

                    if text_content is None:
                        logger.warning(
                            f"Could not find text content key in sentence dict: {sentence_dict}")
                        continue

                    # Create a new standardized dict or modify in place
                    standardized_sentence = sentence_dict.copy()  # Work on a copy
                    if found_key != "text":  # Standardize to 'text'
                        standardized_sentence["text"] = standardized_sentence.pop(
                            found_key)

                    standardized_transcript.append(standardized_sentence)

                # Use the standardized list for further validation
                transcript = standardized_transcript
                # --- End Key Name Standardization ---

            elif isinstance(data, list):
                # If the top level is a list, assume it's the transcript directly
                logger.debug("Parsed transcript response directly as list.")
                transcript = data
                # Apply standardization here too if necessary
                standardized_transcript = []
                for sentence_dict in transcript:
                    # ... (add standardization logic similar to above) ...
                    standardized_transcript.append(standardized_sentence)
                transcript = standardized_transcript

            else:
                logger.warning(
                    f"Unexpected JSON structure for transcript: {type(data)}. Expecting list or dict.")
                return []

            # Validate each sentence
            valid_sentences = []
            for i, sentence in enumerate(transcript):
                if not isinstance(sentence, dict):
                    logger.warning(f"Skipping sentence {i}: not a dictionary")
                    continue
                # --- Make sure the key here matches your prompt ('text' or 'transcript') ---
                text_key = "transcript" if "transcript" in sentence else "text"
                if text_key not in sentence:
                    logger.warning(
                        f"Skipping sentence {i}: missing '{text_key}' field")
                    continue
                # Standardize to 'text' key internally if needed
                if text_key != "text":
                    sentence["text"] = sentence.pop(text_key)
                # --- End Key Check ---

                if "start_time" not in sentence:
                    logger.warning(
                        f"Sentence {i} missing 'start_time', defaulting to 0.0")
                    sentence["start_time"] = 0.0
                if "end_time" not in sentence:
                    logger.warning(
                        f"Sentence {i} missing 'end_time', defaulting to 0.0")
                    sentence["end_time"] = 0.0

                # Ensure times are floats
                try:
                    sentence["start_time"] = float(sentence["start_time"])
                    sentence["end_time"] = float(sentence["end_time"])
                except (ValueError, TypeError):
                    logger.warning(
                        f"Sentence {i} has non-numeric time values ({sentence.get('start_time')}, {sentence.get('end_time')}). Skipping.")
                    continue

                valid_sentences.append(sentence)

            logger.info(
                f"Successfully parsed {len(valid_sentences)} transcript sentences")
            return valid_sentences

        except json.JSONDecodeError as e:
            logger.error(
                f"Invalid JSON format in Gemini response (transcript): {e}")
            # Log raw response on error
            logger.debug(
                f"Raw Gemini response (transcript parse): {gemini_response}")
            return []
        except Exception as e:
            logger.error(f"Error parsing transcript: {e}", exc_info=True)
            return []

    def _parse_scene_boundaries(self, gemini_response_text: str) -> List[Tuple[float, float]]:
        """
        Parses scene boundaries from the Gemini API response string.
        Expects a JSON structure, potentially with a top-level 'scene_boundaries' key.

        Args:
            gemini_response_text: The raw text response from the Gemini API.

        Returns:
            A list of tuples, where each tuple represents a scene boundary (start, end).
        """
        if not gemini_response_text or not gemini_response_text.strip():
            logger.warning(
                "Received empty Gemini response for scene boundaries.")
            return []
        try:
            # Clean potential markdown fences
            cleaned_text = gemini_response_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:-3].strip()
            elif cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:-3].strip()

            if not cleaned_text:
                logger.warning(
                    "Gemini response for boundaries became empty after cleaning.")
                return []

            data = json.loads(cleaned_text)

            boundaries_list = []
            # Check if boundaries are directly a list or nested under a key
            if isinstance(data, dict) and "scene_boundaries" in data and isinstance(data["scene_boundaries"], list):
                boundaries_list = data["scene_boundaries"]
            elif isinstance(data, list):
                # If the top level is a list, assume it's the boundaries list directly
                # This might need adjustment if Gemini returns events differently
                boundaries_list = data
                logger.debug(
                    "Parsed scene boundaries directly from top-level list.")
            else:
                logger.warning(
                    f"Unexpected JSON structure for scene boundaries: {type(data).__name__}. Looking for list or dict with 'scene_boundaries' key.")
                return []

            # Validate items in the list
            valid_boundaries = []
            for item in boundaries_list:
                start, end = None, None
                if isinstance(item, dict) and all(k in item for k in ["start_time", "end_time"]):
                    start, end = item.get(
                        "start_time"), item.get("end_time")
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    start, end = item[0], item[1]

                if start is not None and end is not None:
                    try:
                        start_f = float(start)
                        end_f = float(end)
                        if end_f > start_f:  # Ensure valid range
                            valid_boundaries.append((start_f, end_f))
                        else:
                            logger.warning(
                                f"Skipping scene boundary with invalid time range: start={start_f}, end={end_f}")
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Skipping scene boundary with non-numeric time format: {item}")
                else:
                    logger.warning(
                        f"Skipping invalid scene boundary item structure: {item}")

            logger.info(
                f"Successfully parsed {len(valid_boundaries)} scene boundaries.")
            # Sort boundaries by start time
            valid_boundaries.sort(key=lambda x: x[0])
            return valid_boundaries

        except json.JSONDecodeError as e:
            logger.error(
                f"Invalid JSON format in Gemini response for scene boundaries: {e}")
            logger.debug(
                f"Raw Gemini response (boundaries parse): {gemini_response_text}")
            return []
        except Exception as e:
            logger.error(
                f"Error parsing scene boundaries: {e}", exc_info=True)
            return []

    def _save_segment_analysis_to_json(self, segment_path: str, analysis_results: dict) -> None:
        """
        Saves segment analysis results to a JSON file next to the segment file.
        
        Args:
            segment_path: Path to the video segment file.
            analysis_results: Dictionary containing the analysis results.
        """
        try:
            # Generate the JSON filename based on the segment filename
            segment_dir = os.path.dirname(segment_path)
            segment_basename = os.path.basename(segment_path)
            segment_name = os.path.splitext(segment_basename)[0]  # Remove extension
            json_filename = f"{segment_name}_analysis.json"
            json_path = os.path.join(segment_dir, json_filename)
            
            # Ensure the directory exists
            os.makedirs(segment_dir, exist_ok=True)
            
            # Add timestamp to the analysis results
            analysis_with_metadata = analysis_results.copy()
            analysis_with_metadata["segment_path"] = segment_path
            analysis_with_metadata["analysis_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Write the analysis results to the JSON file
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(analysis_with_metadata, f, indent=4, ensure_ascii=False)
                
            logger.info(f"Saved segment analysis to {json_path}")
            
        except Exception as e:
            logger.error(f"Failed to save segment analysis to JSON: {e}", exc_info=True)
    
    def _parse_gemini_analysis(self, gemini_response_text: str) -> dict:
        """
        Parses the analysis results (summary, topics) from the Gemini API response string.
        Expects a JSON object with 'summary' and 'topics' keys.

        Args:
            gemini_response_text: The raw text response from the Gemini API.

        Returns:
            A dictionary containing the parsed analysis results.
        """
        if not gemini_response_text or not gemini_response_text.strip():
            logger.warning(
                "Received empty Gemini response for segment analysis.")
            return {"summary": "No analysis available", "topics": []}

        try:
            # Clean potential markdown fences
            cleaned_text = gemini_response_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:-3].strip()
            elif cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:-3].strip()

            if not cleaned_text:
                logger.warning(
                    "Gemini response for analysis became empty after cleaning.")
                return {"summary": "No analysis available (empty after cleaning)", "topics": []}

            data = json.loads(cleaned_text)

            if not isinstance(data, dict):
                logger.error(
                    f"Expected JSON object for analysis, got {type(data).__name__}")
                return {"summary": "Invalid API response format", "topics": []}

            # Use .get() for safety and provide defaults
            summary = data.get("summary", "No summary provided.")
            topics = data.get("topics", [])

            # Ensure topics is a list of strings
            if not isinstance(topics, list):
                logger.warning(
                    f"Expected 'topics' to be a list, got {type(topics).__name__}. Converting.")
                topics = [str(topics)]  # Attempt conversion
            else:
                # Ensure all items are strings
                topics = [str(t) for t in topics]

            analysis = {
                "summary": summary,
                "topics": topics
            }
            return analysis

        except json.JSONDecodeError as e:
            logger.error(
                f"Invalid JSON format in Gemini analysis response: {e}")
            logger.info(
                f"Raw Gemini response (analysis parse): {gemini_response_text}")
            return {"summary": "Invalid API response", "topics": [], "error": f"JSONDecodeError: {e}"}
        except Exception as e:
            logger.error(f"Error parsing Gemini analysis: {e}", exc_info=True)
            return {"summary": "Analysis parsing failed", "topics": [], "error": str(e)}

    # --- Segmentation and Analysis Methods ---

    def _detect_silent_periods(self, scene_transcript: List[dict]) -> List[Tuple[float, float]]:
        """
        Detects silent periods within the scene's transcript, using a basic adaptive threshold.

        Args:
            scene_transcript: The transcript of the scene.

        Returns:
            A list of tuples, where each tuple represents a silent period (start, end).
        """
        silent_periods: List[Tuple[float, float]] = []

        if not scene_transcript or len(scene_transcript) < 2:
            logger.debug(
                "Not enough transcript items to detect silent periods.")
            return silent_periods

        time_gaps: List[float] = []
        for i in range(1, len(scene_transcript)):
            try:
                start_time = float(scene_transcript[i].get("start_time", 0.0))
                prev_end_time = float(
                    scene_transcript[i - 1].get("end_time", 0.0))
                gap = start_time - prev_end_time
                if gap > 0.1:  # Only consider gaps greater than 100ms as potential silence
                    time_gaps.append(gap)
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid time format encountered while calculating gaps between transcript items {i-1} and {i}.")
                continue

        if not time_gaps:
            logger.debug(
                "No significant positive time gaps found between transcript items.")
            return silent_periods

        # Calculate adaptive threshold (e.g., mean + std dev, or just mean)
        avg_gap = sum(time_gaps) / len(time_gaps)
        # e.g., 120% of average, but at least 1 second
        SILENCE_THRESHOLD = max(avg_gap * 1.2, 1.0)

        logger.debug(
            f"Calculated silence threshold: {SILENCE_THRESHOLD:.2f}s (based on {len(time_gaps)} gaps, avg={avg_gap:.2f}s)")

        for i in range(1, len(scene_transcript)):
            try:
                prev_item = scene_transcript[i - 1]
                curr_item = scene_transcript[i]
                prev_end_time = float(prev_item.get("end_time", 0.0))
                curr_start_time = float(curr_item.get("start_time", 0.0))
                time_gap = curr_start_time - prev_end_time

                if time_gap > SILENCE_THRESHOLD:
                    silent_periods.append((prev_end_time, curr_start_time))
            except (ValueError, TypeError):
                continue

        logger.info(
            f"Detected {len(silent_periods)} potential silent periods based on threshold {SILENCE_THRESHOLD:.2f}s.")
        return silent_periods

    def _align_timestamps_and_segment(
        self,
        video_path: str,
        output_dir: str,
        scene_start: Optional[float],
        scene_end: Optional[float],
        silent_periods: List[Tuple[float, float]],
    ) -> List[Tuple[float, float, str]]:
        """
        Determines segment boundaries based on scene limits and silent periods,
        aligns them to keyframes, and segments the video.

        Args:
            video_path: Path to the input video.
            output_dir: Directory to save video segments.
            scene_start: Start time of the scene (absolute video time).
            scene_end: End time of the scene (absolute video time).
            silent_periods: List of silent periods (start, end) (absolute video time).

        Returns:
            A list of tuples: (aligned_start, aligned_end, segment_path).
        """
        segment_timestamps: List[Tuple[float, float]] = []
        segment_paths_result: List[Tuple[float, float, str]] = []

        # Determine the absolute start and end points for segmentation
        start_point = scene_start if scene_start is not None else 0.0
        end_point = scene_end

        # If scene_end is None, get video duration
        if end_point is None:
            logger.info(
                "Scene end time is None. Attempting to get video duration.")
            try:
                with VideoFileClip(video_path) as clip:
                    end_point = clip.duration
                if end_point is None or end_point <= start_point:
                    logger.error(
                        f"Could not determine valid video duration for {video_path}. Cannot segment.")
                    return []
                logger.info(
                    f"Using video duration ({end_point:.2f}s) as end point.")
            except Exception as e:
                logger.error(
                    f"Failed to get video duration using moviepy: {e}")
                return []

        # Create initial segment boundaries based on silence
        current_segment_start = start_point
        if not silent_periods:
            if end_point > start_point:
                segment_timestamps.append((start_point, end_point))
            logger.info(
                "No silent periods detected, creating one segment for the range.")
        else:
            silent_periods.sort(key=lambda x: x[0])
            for silence_start, silence_end in silent_periods:
                silence_start = max(silence_start, start_point)
                silence_end = min(silence_end, end_point)
                if silence_start > current_segment_start:
                    segment_timestamps.append(
                        (current_segment_start, silence_start))
                current_segment_start = max(silence_end, current_segment_start)
            if current_segment_start < end_point:
                segment_timestamps.append((current_segment_start, end_point))
            logger.info(
                f"Generated {len(segment_timestamps)} potential segment timestamps based on silence.")

        # Filter out very short segments
        MIN_SEGMENT_DURATION = 1.0
        filtered_timestamps = []
        for start, end in segment_timestamps:
            if end - start >= MIN_SEGMENT_DURATION:
                filtered_timestamps.append((start, end))
            else:
                logger.debug(
                    f"Skipping very short segment: {start:.2f}s - {end:.2f}s")

        if not filtered_timestamps:
            logger.warning(
                "No valid segment timestamps remained after filtering short durations.")
            return []
        logger.info(
            f"{len(filtered_timestamps)} segments remained after filtering.")

        # Align timestamps to keyframes
        aligned_timestamps = video_segmentation.align_timestamps_to_keyframes(
            video_path, filtered_timestamps
        )
        logger.info(
            f"Aligned {len(aligned_timestamps)} timestamps to keyframes.")

        # Segment the video
        timestamp_suffix = time.strftime("%Y%m%d%H%M%S")
        segment_name_template = f"segment_{timestamp_suffix}_$INDEX.mp4"
        segment_paths = video_segmentation.segment_video(
            video_path, output_dir, aligned_timestamps, output_name_template=segment_name_template
        )
        logger.info(
            f"Created {len(segment_paths)} segment files in {output_dir}.")

        # Create the result list
        if len(aligned_timestamps) != len(segment_paths):
            logger.warning(
                f"Mismatch between aligned timestamps ({len(aligned_timestamps)}) and created segment files ({len(segment_paths)}). Results may be incomplete.")

        for i, (start, end) in enumerate(aligned_timestamps):
            if i < len(segment_paths):
                segment_paths_result.append((start, end, segment_paths[i]))
            else:
                logger.error(
                    f"Missing segment file for timestamp range: {start}-{end}")
                segment_paths_result.append((start, end, "ERROR_PATH_MISSING"))

        return segment_paths_result

    def _analyze_segment(self, segment_path: str) -> dict:
        """
        Analyzes a video segment using the Gemini API, parses the result,
        and saves the analysis as a JSON file next to the segment file.

        Args:
            segment_path: Path to the video segment file.

        Returns:
            A dictionary containing the analysis results (summary, topics).
        """
        logger.info(f"Analyzing segment: {segment_path}")
        try:
            # Construct the prompt for Gemini - ask for JSON
            prompt = (
                "Summarize this video segment and identify the key topics discussed. "
                "Respond ONLY with a JSON object containing 'summary' and 'topics' (list of strings) keys."
            )

            # Send the video segment to the Gemini API for analysis
            gemini_response_text = self.gemini_client.analyze(
                prompt, segment_path)

            # Parse the Gemini API response
            analysis_results = self._parse_gemini_analysis(
                gemini_response_text)
            
            # Save the analysis results to a JSON file next to the segment file
            self._save_segment_analysis_to_json(segment_path, analysis_results)
            
            return analysis_results

        except Exception as e:
            logger.error(
                f"Error analyzing segment {segment_path}: {e}", exc_info=True)
            # Return a consistent error structure
            error_results = {"summary": "Analysis failed", "topics": [], "error": str(e)}
            
            # Try to save the error results to JSON as well
            try:
                self._save_segment_analysis_to_json(segment_path, error_results)
            except Exception as save_err:
                logger.error(f"Failed to save error results to JSON: {save_err}")
                
            return error_results
