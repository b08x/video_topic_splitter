# processing/video/scene_detection.py
#!/usr/bin/env python3
"""Scene detection, frame extraction, and video splitting functionality.

This module leverages the PySceneDetect library to perform common video analysis
tasks related to identifying distinct scenes within a video file. It provides
functions to:
1. Detect scene boundaries using content-aware algorithms.
2. Extract representative thumbnail frames from each detected scene.
3. Split the original video into separate files, one for each scene.

Dependencies:
    - PySceneDetect (scenedetect): Core library for scene analysis.
    - FFmpeg: Required by PySceneDetect for video splitting functionality
      (`split_video_ffmpeg`). Ensure FFmpeg is installed and accessible in the
      system's PATH.
"""

import csv
import logging
import os
from typing import Dict, List, Optional, Tuple

# PySceneDetect imports
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector, AdaptiveDetector
from scenedetect.scene_manager import save_images  # Keep for frame extraction
from scenedetect.stats_manager import StatsManager
from scenedetect.video_splitter import split_video_ffmpeg  # Import for splitting

logger = logging.getLogger(__name__)


def detect_scenes(
    video_path: str,
    output_dir: str,  # Base output dir for scene-related files (CSV, frames)
    threshold: float = 27.0,
    min_scene_len_sec: float = 1.0,
    save_csv: bool = True,
    # Videos <= this length are considered "short"
    short_video_threshold_sec: float = 60.0,
) -> List[Tuple[float, float]]:
    """Detects scenes in a video using PySceneDetect's detectors.

    This function analyzes the video specified by `video_path` to identify
    significant changes between frames, marking the boundaries of distinct scenes.
    It primarily uses the `ContentDetector`, which detects changes in visual content.
    If `ContentDetector` fails to find any scenes, it falls back to using the
    `AdaptiveDetector` as a secondary strategy.

    For videos considered "short" (duration <= `short_video_threshold_sec`) where
    neither detector finds scene cuts, a single scene spanning the entire video
    duration is generated. This ensures that even short clips or single-shot
    videos produce a valid scene boundary output.

    Args:
        video_path: Absolute or relative path to the input video file.
        output_dir: Path to the directory where scene-related output files
            (e.g., `scenes.csv`) will be saved. The directory will be
            created if it does not exist.
        threshold: Detection threshold for the `ContentDetector`. Lower values
            result in detecting more, potentially shorter, scenes (i.e., higher
            sensitivity). Higher values require more significant changes to
            trigger a scene cut. Defaults to 27.0.
        min_scene_len_sec: The minimum duration (in seconds) a sequence of frames
            must have to be considered a distinct scene. Scenes shorter than this
            will be merged with adjacent scenes. Defaults to 1.0 second.
        save_csv: If True, a CSV file named `scenes.csv` containing detailed
            information about each detected scene (start/end times, frames,
            duration) will be saved in the `output_dir`. Defaults to True.
        short_video_threshold_sec: The maximum duration in seconds for a video
            to be classified as "short". If a video is shorter than or equal to
            this duration and no scenes are detected by either detector, a single
            scene covering the entire video is returned. Defaults to 60.0 seconds.

    Returns:
        A list of tuples. Each tuple represents a detected scene and contains
        the start time and end time of the scene in seconds.
        Example: `[(0.0, 15.5), (15.5, 45.2), (45.2, 60.0)]`
        Returns an empty list if the video could not be processed or if no scenes
        were detected (and the video is longer than `short_video_threshold_sec`).

    Raises:
        FileNotFoundError: If the `video_path` does not point to an existing file.
        ValueError: If the video file has an invalid or zero framerate.
        RuntimeError: If an unexpected error occurs during the PySceneDetect
            processing pipeline (e.g., issues opening the video, detector errors).
            The original exception is chained.

    Notes:
        - This function relies on the PySceneDetect library.
        - The choice of detector and parameters (`threshold`, `min_scene_len_sec`)
          can significantly impact the results. Experimentation may be needed
          for optimal performance on specific types of video content.
    """
    scene_boundaries_sec: List[Tuple[float, float]] = []
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    video = None  # Initialize video object
    try:
        # Open video using PySceneDetect
        video = open_video(video_path)
        fps = video.frame_rate
        if not fps or fps <= 0:
            raise ValueError("Invalid or zero framerate detected.")

        # Get video duration in seconds
        video_duration_sec = video.duration.get_seconds()
        logger.info(f"Video duration: {video_duration_sec:.2f} seconds")

        # Convert min_scene_len from seconds to frames
        min_scene_len_frames = int(min_scene_len_sec * fps)

        # --- Attempt 1: ContentDetector ---
        stats_manager_content = StatsManager()
        scene_manager_content = SceneManager(stats_manager_content)
        scene_manager_content.add_detector(
            ContentDetector(threshold=threshold,
                            min_scene_len=min_scene_len_frames)
        )

        logger.info(
            "Detecting scenes in %s (ContentDetector: threshold=%.1f, min_len=%.2f sec)...",
            video_path,
            threshold,
            min_scene_len_sec,
        )
        scene_manager_content.detect_scenes(video=video, show_progress=False)
        scene_list_timecodes = scene_manager_content.get_scene_list()

        # --- Attempt 2: AdaptiveDetector (Fallback) ---
        if not scene_list_timecodes:
            logger.info(
                "No scenes detected with ContentDetector. Trying AdaptiveDetector...")
            video.reset()  # Reset video position for the next detector

            stats_manager_adaptive = StatsManager()
            scene_manager_adaptive = SceneManager(stats_manager_adaptive)
            # Add AdaptiveDetector with potentially sensitive parameters
            scene_manager_adaptive.add_detector(
                AdaptiveDetector(
                    adaptive_threshold=3.0,  # Default is 3.0, lower is more sensitive
                    min_scene_len=min_scene_len_frames,
                    # Consider adjusting other AdaptiveDetector params if needed
                )
            )

            logger.info("Detecting scenes with AdaptiveDetector...")
            scene_manager_adaptive.detect_scenes(
                video=video, show_progress=False)
            scene_list_timecodes = scene_manager_adaptive.get_scene_list()

            if scene_list_timecodes:
                logger.info("AdaptiveDetector found %d scenes.",
                            len(scene_list_timecodes))
            else:
                logger.warning(
                    "No scenes detected with AdaptiveDetector either.")

                # --- Handle Short Videos with No Detected Scenes ---
                if video_duration_sec <= short_video_threshold_sec:
                    logger.info(
                        f"Short video detected ({video_duration_sec:.2f} sec <= "
                        f"{short_video_threshold_sec:.1f} sec). Creating a single scene."
                    )
                    start_tc = video.base_timecode
                    end_tc = video.base_timecode + \
                        int(video_duration_sec * fps)
                    # Ensure end_tc is at least one frame after start_tc if duration is very small
                    if end_tc.get_frames() <= start_tc.get_frames():
                        end_tc = start_tc + 1
                    scene_list_timecodes = [(start_tc, end_tc)]
                    logger.info("Created 1 scene spanning the entire video.")

        # --- Process Results ---
        # Convert Timecode objects to seconds for the final output
        scene_boundaries_sec = [
            (start.get_seconds(), end.get_seconds())
            for start, end in scene_list_timecodes
        ]

        logger.info("Detected %d scenes.", len(scene_boundaries_sec))

        # --- Save CSV Output ---
        if save_csv and scene_boundaries_sec:
            csv_path = os.path.join(output_dir, "scenes.csv")
            try:
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "Scene",
                            "Start Frame",
                            "End Frame",
                            "Start Time (sec)",
                            "End Time (sec)",
                            "Duration (sec)",
                        ]
                    )
                    # Use the timecode list for accurate frame numbers
                    for i, (start_tc, end_tc) in enumerate(scene_list_timecodes):
                        start_time_sec, end_time_sec = scene_boundaries_sec[i]
                        duration_sec = end_time_sec - start_time_sec
                        writer.writerow(
                            [
                                i + 1,  # 1-based scene number
                                start_tc.get_frames(),
                                end_tc.get_frames(),
                                f"{start_time_sec:.3f}",
                                f"{end_time_sec:.3f}",
                                f"{duration_sec:.3f}",
                            ]
                        )
                logger.info("Scene list saved to %s", csv_path)
            except IOError as e_save:
                logger.error(
                    f"Error writing scene list to CSV file {csv_path}: {e_save}")
            except Exception as e_save:
                logger.error(
                    f"Unexpected error saving scene list to CSV: {e_save}")

        return scene_boundaries_sec

    except FileNotFoundError:
        logger.error("Video file not found: %s", video_path)
        raise  # Re-raise the FileNotFoundError
    except ValueError as e_val:  # Catch specific framerate error
        logger.error("Video processing error for %s: %s", video_path, e_val)
        raise
    except Exception as e:
        logger.error(
            "Error during scene detection for %s: %s", video_path, str(e), exc_info=True
        )
        # Wrap the original exception for better context
        raise RuntimeError(
            f"Scene detection failed for {video_path}: {str(e)}") from e
    finally:
        # Ensure video file handle is released
        if video and hasattr(video, 'release'):
            try:
                video.release()
            except Exception as e_release:
                logger.warning(
                    f"Error releasing video handle for {video_path}: {e_release}"
                )


def extract_scene_frames(
    video_path: str,
    scene_boundaries: List[Tuple[float, float]],
    output_dir: str,  # Dir where frame images will be saved
    num_frames_per_scene: int = 1,
    frame_format: str = "jpg",
    jpg_quality: int = 90,
    short_video_frames: int = 3,  # Number of frames to extract for short videos
) -> List[Dict]:
    """Extracts representative frames (thumbnails) from specified video scenes.

    This function iterates through the provided `scene_boundaries` (typically
    generated by `detect_scenes`) and extracts one or more frames from each
    scene interval in the `video_path`. The extracted frames are saved as image
    files (JPG or PNG) in the `output_dir`.

    A special case exists for videos identified as having a single scene that
    spans a short duration (currently hardcoded check, ideally aligned with
    `detect_scenes` logic if possible): instead of extracting `num_frames_per_scene`,
    it extracts `short_video_frames` to provide more visual context for short clips.

    Args:
        video_path: Absolute or relative path to the input video file.
        scene_boundaries: A list of tuples, where each tuple defines a scene
            with its start and end time in seconds. Example: `[(0.0, 15.5), (15.5, 45.2)]`.
            This list is typically the output of the `detect_scenes` function.
        output_dir: Path to the directory where the extracted frame image files
            will be saved. The directory will be created if it does not exist.
            Frames will be named using the pattern 'Scene#-Image#.ext'.
        num_frames_per_scene: The target number of frames to extract from each
            scene. Frames are typically chosen evenly spaced within the scene's
            duration. Defaults to 1 (extracting roughly the middle frame).
        frame_format: The desired image format for the extracted frames.
            Supported formats typically include 'jpg' and 'png'. Defaults to "jpg".
        jpg_quality: If `frame_format` is 'jpg', this specifies the JPEG quality
            level (1-100, higher is better quality). Ignored for other formats.
            Defaults to 90.
        short_video_frames: The number of frames to extract if the video consists
            of a single scene and is considered "short" (duration <= 60 seconds).
            This overrides `num_frames_per_scene` in that specific case. Defaults to 3.

    Returns:
        A list of dictionaries, one for each scene processed. Each dictionary
        contains information about the scene and the paths to the extracted frames:
            - 'scene_id' (int): 1-based index of the scene.
            - 'start_time' (float): Start time of the scene in seconds.
            - 'end_time' (float): End time of the scene in seconds.
            - 'duration' (float): Duration of the scene in seconds.
            - 'frame_paths' (List[str]): A list of absolute paths to the image
              files extracted for this scene.
        Returns an empty list if `scene_boundaries` is empty or if an error occurs
        during processing.

    Raises:
        FileNotFoundError: If the `video_path` does not point to an existing file.
        ValueError: If `scene_boundaries` is empty, or if the video file has an
            invalid or zero framerate.
        RuntimeError: If an unexpected error occurs during frame extraction, often
            related to PySceneDetect's `save_images` function or video decoding.
            The original exception is chained.

    Notes:
        - Relies on PySceneDetect's `save_images` utility function.
        - The exact frames extracted depend on PySceneDetect's internal logic for
          selecting representative frames based on `num_frames_per_scene`.
        - The check for "short video with single scene" uses a fixed threshold (60s)
          which might need adjustment based on how `detect_scenes` defines short videos.
    """
    if not scene_boundaries:
        logger.warning(
            "No scene boundaries provided for frame extraction. Returning empty list."
        )
        # Raise ValueError instead? Depends on desired strictness.
        # raise ValueError("Scene boundaries list cannot be empty for frame extraction.")
        return []

    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    video = None
    scene_info = []  # Initialize here for broader scope

    try:
        video = open_video(video_path)
        fps = video.frame_rate
        if not fps or fps <= 0:
            raise ValueError(
                "Invalid or zero framerate detected for frame extraction.")

        # Determine number of frames to extract based on video type
        num_frames_to_extract = num_frames_per_scene
        if len(scene_boundaries) == 1:
            start_time, end_time = scene_boundaries[0]
            duration = end_time - start_time
            # Use a consistent threshold, maybe pass short_video_threshold_sec here?
            # For now, using a hardcoded 60.0 matching the default in detect_scenes
            short_video_threshold_check = 60.0
            if duration <= short_video_threshold_check:
                is_short_single_scene = True
                logger.info(
                    f"Detected short video with single scene ({duration:.2f} sec <= "
                    f"{short_video_threshold_check:.1f} sec). Will extract {short_video_frames} frames."
                )
                num_frames_to_extract = short_video_frames
            # else: num_frames_to_extract remains num_frames_per_scene

        # Convert scene boundaries (seconds) back to PySceneDetect Timecode objects
        scene_list_timecodes = []
        for start_sec, end_sec in scene_boundaries:
            # Calculate frame numbers based on video's base timecode and framerate
            start_frame = int(start_sec * fps)
            end_frame = int(end_sec * fps)
            start_tc = video.base_timecode + start_frame
            end_tc = video.base_timecode + end_frame

            # Ensure end frame is at least one frame after start frame
            # PySceneDetect might handle this, but being explicit can prevent errors
            if end_tc.get_frames() <= start_tc.get_frames():
                logger.warning(
                    f"Scene ending at {end_sec:.3f}s has end frame ({end_frame}) <= start frame ({start_frame}). Adjusting end frame."
                )
                end_tc = start_tc + 1  # Ensure minimum 1 frame duration for timecode pair

            scene_list_timecodes.append((start_tc, end_tc))

        logger.info(
            "Extracting %d frame(s) per scene from %d scenes into %s...",
            num_frames_to_extract,
            len(scene_list_timecodes),
            output_dir,
        )

        # Use PySceneDetect's save_images function
        # Note: save_images uses 0-based scene indices internally for its dictionary keys
        image_filenames_dict = save_images(
            scene_list=scene_list_timecodes,
            video=video,
            num_images=num_frames_to_extract,
            output_dir=output_dir,
            image_extension=frame_format,
            encoder_param=jpg_quality if frame_format.lower() == 'jpg' else None,
            # Naming template uses 1-based scene number ($SCENE_NUMBER)
            image_name_template='$SCENE_NUMBER-$IMAGE_NUMBER',
            show_progress=False,  # Keep logs cleaner
        )

        # Structure the output with absolute paths and scene details
        total_frames_extracted = 0
        for i, (start_time, end_time) in enumerate(scene_boundaries):
            # Key for image_filenames_dict is the 0-based index `i`
            relative_frame_paths = image_filenames_dict.get(i, [])
            absolute_frame_paths = [
                os.path.join(output_dir, f) for f in relative_frame_paths
            ]
            total_frames_extracted += len(absolute_frame_paths)

            scene_info.append(
                {
                    "scene_id": i + 1,  # User-facing scene ID is 1-based
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                    "frame_paths": absolute_frame_paths,
                }
            )
            if not absolute_frame_paths:
                logger.warning(
                    f"No frames were extracted for scene {i+1} ({start_time:.2f}s - {end_time:.2f}s)."
                )

        logger.info(
            "Extracted %d total frames from %d scenes.",
            total_frames_extracted,
            len(scene_boundaries),
        )
        return scene_info

    except FileNotFoundError:
        logger.error(
            "Video file not found for frame extraction: %s", video_path)
        raise
    except ValueError as e_val:  # Catch specific framerate/boundary errors
        logger.error(
            "Invalid input for frame extraction from %s: %s", video_path, e_val
        )
        raise
    except Exception as e:
        logger.error(
            "Error extracting scene frames from %s: %s", video_path, str(e), exc_info=True
        )
        raise RuntimeError(
            f"Frame extraction failed for {video_path}: {str(e)}") from e
    finally:
        # Ensure video file handle is released
        if video and hasattr(video, 'release'):
            try:
                video.release()
            except Exception as e_release:
                logger.warning(
                    f"Error releasing video handle during frame extraction for {video_path}: {e_release}"
                )


def _process_csv_row(row: Dict[str, str], row_num: int, csv_path: str, start_col_name: str, end_col_name: str, scenes_list: List[Tuple[float, float]], logger: logging.Logger):
    """Helper function to process a single row from the scenes CSV."""
    start_time_str, end_time_str = None, None  # Initialize for error logging
    try:
        start_time_str = row[start_col_name]
        end_time_str = row[end_col_name]
        start_time = float(start_time_str)
        end_time = float(end_time_str)
        if end_time > start_time:
            scenes_list.append((start_time, end_time))
        else:
            logger.warning(
                f"Skipping invalid time range in row {row_num} of {csv_path}: start={start_time}, end={end_time}")
    except KeyError as e:
        logger.warning(
            f"Missing expected key {e} in row {row_num} of {csv_path}. Skipping row. Row data: {row}")
    except (ValueError, TypeError) as e:
        start_log = f"'{start_time_str}'" if start_time_str is not None else "MISSING"
        end_log = f"'{end_time_str}'" if end_time_str is not None else "MISSING"
        logger.warning(
            f"Invalid numeric value in row {row_num} of {csv_path} (Start: {start_log}, End: {end_log}). Error: {e}. Skipping row.")


def read_scenes_from_csv(csv_path: str) -> List[Tuple[float, float]]:
    """
    Reads scene start and end times from a CSV file generated by PySceneDetect.

    Handles CSVs with or without the initial PySceneDetect version line.
    Expects a header row with columns 'Start Time (sec)' and 'End Time (sec)'. # <-- Updated expectation description

    Args:
        csv_path: Path to the scenes CSV file.

    Returns:
        A list of tuples, where each tuple is (start_time_sec, end_time_sec).
        Returns an empty list if the file cannot be read or parsed correctly.

    Raises:
        FileNotFoundError: If the csv_path does not exist.
        ValueError: If required columns are missing or time values are invalid.
        Exception: For other potential CSV reading errors.
    """
    if not os.path.exists(csv_path):
        logger.error(f"Scene CSV file not found: {csv_path}")
        raise FileNotFoundError(f"Scene CSV file not found: {csv_path}")

    scenes: List[Tuple[float, float]] = []
    # --- CHANGE HERE: Match the actual CSV header format ---
    required_cols_lower = ['start time (sec)', 'end time (sec)']
    # --- END CHANGE ---
    processed = False

    try:
        with open(csv_path, mode='r', encoding='utf-8') as csvfile:
            # Store initial position and read first line
            initial_pos = csvfile.tell()
            first_line = csvfile.readline().strip()
            after_first_line_pos = csvfile.tell()

            # --- Attempt 1: Assume header is on the second line ---
            try:
                # Go to start of second line
                csvfile.seek(after_first_line_pos)
                reader = csv.DictReader(csvfile)
                # Check if fieldnames were successfully read and match expected
                if reader.fieldnames:
                    header = [h.lower().strip() for h in reader.fieldnames]
                    if all(col in header for col in required_cols_lower):
                        logger.debug(
                            f"CSV header found on second line: {reader.fieldnames}")
                        start_col = reader.fieldnames[header.index(
                            required_cols_lower[0])]
                        end_col = reader.fieldnames[header.index(
                            required_cols_lower[1])]
                        # Line 1 version, Line 2 header
                        for row_num, row in enumerate(reader, start=2):
                            _process_csv_row(
                                row, row_num, csv_path, start_col, end_col, scenes, logger)
                        processed = True
                        logger.info(
                            f"Successfully read {len(scenes)} scene boundaries from {csv_path} (header on line 2)")
                    else:
                        logger.debug(
                            f"CSV header not found on second line. Found: {reader.fieldnames}")
                else:
                    logger.debug(
                        "CSV DictReader found no fieldnames assuming header on second line (likely empty file after line 1).")

            except (StopIteration, csv.Error) as e:
                # Handle errors like empty file after first line or malformed CSV
                logger.debug(
                    f"Error/EOF reading CSV assuming header on line 2: {e}. Checking first line.")
            except Exception as e:
                # Catch unexpected errors during processing
                logger.warning(
                    f"Unexpected error processing CSV assuming header on line 2: {e}", exc_info=True)

            # --- Attempt 2: Assume header is on the first line (if Attempt 1 failed) ---
            if not processed:
                # Check if the first line we read looks like the header
                # Use split(',') for robustness against extra spaces around commas
                first_line_cols = [col.strip().lower()
                                   for col in first_line.split(',')]
                if all(col in first_line_cols for col in required_cols_lower):
                    logger.debug(
                        f"CSV header potentially found on first line: {first_line.split(',')}")
                    try:
                        csvfile.seek(initial_pos)  # Reset to beginning
                        # Re-initialize, should use first line
                        reader = csv.DictReader(csvfile)
                        if reader.fieldnames:
                            header = [h.lower().strip()
                                      for h in reader.fieldnames]
                            # Double-check header consistency
                            if not all(col in header for col in required_cols_lower):
                                # This path should ideally not be hit if the first_line_cols check passed,
                                # but it's a safeguard against weird CSV parsing issues.
                                raise ValueError(
                                    f"Inconsistent header detection. First line looked like header '{first_line}', but DictReader got '{reader.fieldnames}'. Required: {required_cols_lower}")

                            start_col = reader.fieldnames[header.index(
                                required_cols_lower[0])]
                            end_col = reader.fieldnames[header.index(
                                required_cols_lower[1])]
                            # Line 1 header
                            for row_num, row in enumerate(reader, start=1):
                                _process_csv_row(
                                    row, row_num, csv_path, start_col, end_col, scenes, logger)
                            processed = True
                            logger.info(
                                f"Successfully read {len(scenes)} scene boundaries from {csv_path} (header on line 1)")
                        else:
                            logger.warning(
                                "CSV DictReader found no fieldnames assuming header on first line (likely empty file).")

                    except (StopIteration, csv.Error, ValueError) as e:
                        logger.error(
                            f"Error reading CSV assuming header on line 1: {e}", exc_info=True)
                        # Don't raise here, let it fall through to the final error if needed
                    except Exception as e:
                        logger.error(
                            f"Unexpected error processing CSV assuming header on line 1: {e}", exc_info=True)

            # --- If neither attempt worked ---
            if not processed:
                logger.error(f"CSV file '{csv_path}' does not contain the required header columns "
                             f"'{required_cols_lower}' on the first or second line, or is improperly formatted.")
                logger.debug(f"First line content: {first_line}")
                # Attempt to read second line again for logging if possible
                try:
                    csvfile.seek(after_first_line_pos)
                    second_line = csvfile.readline().strip()
                    logger.debug(f"Second line content: {second_line}")
                except Exception:
                    logger.debug("Could not read second line for debugging.")
                # Raise a specific error indicating failure
                raise ValueError(
                    f"CSV file '{csv_path}' missing required columns or has unexpected format.")

    except FileNotFoundError:  # Already handled above, but good practice
        raise
    except csv.Error as e:
        # Try to get line number if available
        line_num_info = f" near line {reader.line_num}" if 'reader' in locals(
        ) and hasattr(reader, 'line_num') else ""
        logger.error(
            f"CSV reading error in '{csv_path}'{line_num_info}: {e}", exc_info=True)
        raise Exception(f"CSV reading error in '{csv_path}': {e}") from e
    except Exception as e:
        logger.error(
            f"Failed to read or parse scene CSV '{csv_path}': {e}", exc_info=True)
        raise  # Re-raise other unexpected errors

    # Final check and sort
    if not scenes and processed:  # If processed is true but scenes is empty
        logger.warning(
            f"Successfully parsed CSV '{csv_path}' but extracted 0 valid scene boundaries.")
    elif not scenes and not processed:  # Should have been caught by the ValueError above
        logger.error(
            f"Failed to extract any scene boundaries from {csv_path}.")

    scenes.sort(key=lambda x: x[0])
    return scenes


def split_video_by_scenes(
    video_path: str,
    scene_list: List[Tuple[float, float]],
    output_dir: str,
    output_file_template: str = 'scene_$SCENE_NUMBER.mp4',
    show_progress: bool = True,
    show_output: bool = False,  # Keep ffmpeg logs chatty by default
) -> List[str]:
    """Splits a video into multiple segment files based on detected scene boundaries.

    This function takes a list of scene start and end times (in seconds) and uses
    FFmpeg (via PySceneDetect's `split_video_ffmpeg` helper) to cut the original
    `video_path` into separate video files, one for each scene. The resulting
    video segments are saved in the `output_dir`.

    An optimization exists: if `scene_list` contains only one scene that spans
    nearly the entire duration of the original video, the splitting process is
    skipped, and the path to the original video file is returned, avoiding
    unnecessary processing.

    Args:
        video_path: Absolute or relative path to the input video file to be split.
        scene_list: A list of tuples, where each tuple defines a scene with its
            start and end time in seconds. Example: `[(0.0, 15.5), (15.5, 45.2)]`.
            This list is typically the output of the `detect_scenes` function.
        output_dir: Path to the directory where the split video segment files
            will be saved. The directory will be created if it does not exist.
        output_file_template: A template string for naming the output segment files.
            It should include `$SCENE_NUMBER` which will be replaced by the
            1-based index of the scene. The file extension determines the output
            container format (e.g., '.mp4', '.mkv').
            Defaults to 'scene_$SCENE_NUMBER.mp4'.
        show_progress: If True, displays FFmpeg's progress indicators in the
            standard output/error streams during the splitting process.
            Defaults to False.
        show_output: If True, will show output from ffmpeg for first split.

    Returns:
        A list of strings, where each string is the absolute path to a created
        video segment file. If splitting was skipped due to a single, full-duration
        scene, the list will contain only the original `video_path`. Returns an
        empty list if `scene_list` was empty or if the splitting process failed
        to create any valid files.

    Raises:
        FileNotFoundError: If the `video_path` does not point to an existing file.
        ValueError: If the video file has an invalid or zero framerate.
        RuntimeError: If the video splitting process fails. This is often due to
            an underlying FFmpeg error (e.g., invalid arguments, codec issues,
            file permissions). Check FFmpeg logs if `show_output` is True.
            The original exception is chained.

    Notes:
        - Requires FFmpeg to be installed and accessible in the system's PATH.
        - By default, `` attempts to use stream copying
          (`-c copy`) for speed, which preserves the original video and audio
          codecs. If format conversion or re-encoding is needed, additional
          FFmpeg arguments might be required (passed via an `ffmpeg_args` parameter
          to `split_video_ffmpeg`, though not exposed in this function's signature).
        - The check for a single, full-duration scene uses a small time tolerance (0.5s)
          at the start and end to account for potential minor inaccuracies in detection.
    """
    if not scene_list:
        logger.warning(
            "No scenes provided for splitting. Returning empty list.")
        return []

    video = None  # Initialize video object outside try block for finally clause
    try:
        # Check for single scene spanning the whole video (optimization)
        if len(scene_list) == 1:
            start_time, end_time = scene_list[0]
            # Need to open video briefly to get duration for comparison
            temp_video = open_video(video_path)
            video_duration = temp_video.duration.get_seconds()

            # Safely close the video handle - check if release method exists
            if hasattr(temp_video, 'release'):
                try:
                    temp_video.release()
                except Exception as e_release:
                    logger.warning(
                        f"Error releasing temporary video handle: {e_release}"
                    )
            # No need for an else clause - if release() doesn't exist, we just continue

            # Define a small tolerance for start/end times
            time_tolerance = 0.5  # seconds

            if (
                abs(start_time) < time_tolerance
                and abs(end_time - video_duration) < time_tolerance
            ):
                logger.info(
                    "Single scene spans the entire video (duration %.2fs). Skipping unnecessary splitting.",
                    video_duration,
                )
                return [os.path.abspath(video_path)]  # Return absolute path

        # Proceed with splitting for multiple scenes or partial single scene
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        video = open_video(video_path)
        fps = video.frame_rate
        if not fps or fps <= 0:
            raise ValueError(
                "Invalid or zero framerate detected for video splitting."
            )

        # Convert scene boundaries (seconds) back to PySceneDetect Timecode objects
        scene_list_timecodes = []
        for start_sec, end_sec in scene_list:
            start_frame = int(start_sec * fps)
            end_frame = int(end_sec * fps)
            start_tc = video.base_timecode + start_frame
            end_tc = video.base_timecode + end_frame

            # Ensure end frame is valid and at least one frame after start
            if end_tc.get_frames() <= start_tc.get_frames():
                logger.warning(
                    f"Scene ending at {end_sec:.3f}s has end frame ({end_frame}) <= start frame ({start_frame}) during splitting. Adjusting end frame."
                )
                end_tc = start_tc + 1
            scene_list_timecodes.append((start_tc, end_tc))

        logger.info(
            "Splitting video '%s' into %d scenes in directory: %s",
            os.path.basename(video_path),
            len(scene_list_timecodes),
            output_dir,
        )

        # Construct the full path template for output files
        full_output_template = os.path.join(output_dir, output_file_template)

        # Perform the split using PySceneDetect's helper function
        # This function calls the ffmpeg command line tool.
        split_files_relative = split_video_ffmpeg(
            input_video_path=video_path,
            scene_list=scene_list_timecodes,
            output_file_template=full_output_template,  # Use the path constructed above
            show_progress=show_progress,
            show_output=show_output,
            # Example: Add copy codec args explicitly if needed, though it's default
            # ffmpeg_args=['-map', '0', '-c', 'copy']
        )

        # `split_video_ffmpeg` returns paths based on the template.
        # Convert to absolute paths and verify existence/size.
        created_files_abs = []
        expected_count = len(scene_list_timecodes)
        actual_count = 0

        # Generate expected filenames based on the template and scene count,
        # replicating the padding logic used by split_video_ffmpeg.
        # Padding is at least 3 digits, or more if scene count exceeds 999.
        padding = max(3, len(str(expected_count)))
        expected_filenames = [
            full_output_template.replace(
                '$SCENE_NUMBER', str(i + 1).zfill(padding))
            for i in range(expected_count)
        ]

        # Now check the correctly formatted expected filenames
        for file_path in expected_filenames:
            abs_path = os.path.abspath(file_path)
            if os.path.exists(abs_path) and os.path.getsize(abs_path) > 0:
                created_files_abs.append(abs_path)
                actual_count += 1
            else:
                # Log missing/empty files even if show_output was False for ffmpeg
                logger.warning(
                    f"Expected split file not found or is empty: {abs_path}"
                )

        if actual_count != expected_count:
            logger.warning(
                f"Video splitting possibly incomplete: Expected {expected_count} segment files, but found {actual_count} valid files in {output_dir}. Check FFmpeg logs (rerun with show_output=True if needed)."
            )
            # Depending on requirements, could raise an error here:
            # raise RuntimeError(f"Failed to create all expected video segments. Found {actual_count}/{expected_count}.")

        logger.info(
            "Video splitting complete. Created %d segment file(s).", actual_count
        )
        return created_files_abs

    except FileNotFoundError:
        logger.error("Video file not found for splitting: %s", video_path)
        raise
    except ValueError as e_val:  # Catch specific framerate error
        logger.error(
            "Video processing error during splitting of %s: %s", video_path, e_val
        )
        raise
    except Exception as e:
        logger.error(
            "Error during video splitting for %s: %s", video_path, str(e), exc_info=True
        )
        raise RuntimeError(
            f"Video splitting failed for {video_path}: {str(e)}") from e
    finally:
        # Ensure video file handle is released if it has a release method
        if video and hasattr(video, 'release'):
            try:
                video.release()
            except Exception as e_release:
                logger.warning(
                    f"Error releasing video handle during splitting for {video_path}: {e_release}"
                )
        # This else block was misplaced and caused a syntax error. Removed.
        # else:
        #     logger.info("Video splitting complete.")
