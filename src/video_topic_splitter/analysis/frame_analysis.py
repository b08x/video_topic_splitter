#!/usr/bin/env python3
"""Contextual frame analysis functionality.

This module provides the `ContextualFrameAnalyzer` class, which is responsible
for extracting relevant frames from video segments based on transcript timings,
analyzing these frames for visual content (including software detection),
and integrating this visual analysis with contextual information from the
transcript and topic modeling results. It utilizes computer vision techniques
for frame quality assessment and interacts with external APIs (like Gemini)
for deeper visual understanding.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from PIL import Image

from ..api.gemini import analyze_with_gemini
from ..processing.ocr.ocr_detection import detect_software_names
from ..processing.software.software_detection import detect_software_logos

logger = logging.getLogger(__name__)


class ContextualFrameAnalyzer:
    """Handles frame extraction and analysis with transcript context.

    This class encapsulates the logic for selecting, extracting, saving,
    and analyzing video frames associated with specific transcript segments.
    It assesses frame quality, performs software detection (OCR and logo),
    and leverages a multimodal model (Gemini) for contextual visual analysis.

    Attributes:
        video (VideoFileClip): The video file clip object.
        segments (List[Dict]): List of transcript segments with timestamps.
        project_path (str): Path to the project directory.
        screenshots_dir (str): Path to the directory where screenshots are saved.
        software_list (Optional[List[str]]): List of software names for detection.
        logo_db_path (Optional[str]): Path to the logo database.
        ocr_lang (str): Language for OCR detection.
        logo_threshold (float): Confidence threshold for logo detection.
        quality_threshold (float): Minimum quality score for saving screenshots.
        save_format (str): Format for saving screenshots ('jpg' or 'png').
        compression_quality (int): Quality setting for JPEG compression.
        frame_cache (Dict): Cache to store analyzed frame information.
    """

    def __init__(
        self,
        video_path: str,
        transcript_segments: List[Dict],
        project_path: str,
        software_list: Optional[List[str]] = None,
        logo_db_path: Optional[str] = None,
        ocr_lang: str = "eng",
        logo_threshold: float = 0.8,
        quality_threshold: float = 0.5,
        save_format: str = "jpg",
        compression_quality: int = 85,
    ):
        """Initialize the ContextualFrameAnalyzer.

        Args:
            video_path (str): Path to the video file.
            transcript_segments (List[Dict]): List of transcript segments, where
                each segment is a dictionary containing at least 'start_time',
                'end_time', and 'segment_id'.
            project_path (str): Path to the project directory where outputs
                (like screenshots) will be saved.
            software_list (Optional[List[str]], optional): A list of specific
                software names to look for during analysis. Defaults to None.
            logo_db_path (Optional[str], optional): Path to a directory
                containing software logo images for detection. Defaults to None.
            ocr_lang (str, optional): The language code(s) for OCR text
                detection (e.g., 'eng' for English). Defaults to "eng".
            logo_threshold (float, optional): The minimum confidence score
                (0.0 to 1.0) required to consider a logo detection valid.
                Defaults to 0.8.
            quality_threshold (float, optional): The minimum quality score
                (0.0 to 1.0) a frame must have to be saved as a screenshot.
                Defaults to 0.5.
            save_format (str, optional): The image format for saving screenshots
                ('jpg' or 'png'). Defaults to "jpg".
            compression_quality (int, optional): The compression quality for
                saved JPEG images (1-100, higher means better quality and larger
                file size). Ignored for PNG. Defaults to 85.

        Raises:
            FileNotFoundError: If the video_path does not exist.
            Exception: If moviepy fails to load the video file.
        """
        try:
            self.video = VideoFileClip(video_path)
        except Exception as e:
            logger.error(f"Failed to load video file '{video_path}': {e}")
            raise
        self.segments = transcript_segments
        self.project_path = project_path
        self.screenshots_dir = os.path.join(project_path, "screenshots")
        self.software_list = software_list
        self.logo_db_path = logo_db_path
        self.ocr_lang = ocr_lang
        self.logo_threshold = logo_threshold
        self.quality_threshold = quality_threshold
        self.save_format = save_format.lower()
        self.compression_quality = compression_quality
        self.frame_cache = {}  # Cache analyzed frames to avoid reprocessing

        # Create screenshots directory structure
        os.makedirs(self.screenshots_dir, exist_ok=True)
        logger.info(f"Screenshots will be saved in: {self.screenshots_dir}")

    def _select_best_frames(
        self, frames: List[Dict], max_frames: int = 3
    ) -> List[Dict]:
        """Select the best frames from a list based on quality scores.

        Sorts the input frames by their 'quality_score' in descending order
        and selects up to `max_frames`. The selected frames are then returned,
        sorted by their original timestamp to maintain temporal order.

        Args:
            frames (List[Dict]): A list of frame information dictionaries. Each
                dictionary must contain at least 'timestamp' and 'quality_score'.
            max_frames (int, optional): The maximum number of frames to select.
                Defaults to 3.

        Returns:
            List[Dict]: A list containing the selected best frames, sorted by
            timestamp. Returns an empty list if the input `frames` is empty.
        """
        if not frames:
            return []

        # Sort frames by quality score (highest first)
        sorted_frames_by_quality = sorted(
            frames, key=lambda x: x.get("quality_score", 0.0), reverse=True
        )

        # Select the timestamps of the top N frames
        selected_timestamps = {
            f["timestamp"] for f in sorted_frames_by_quality[:max_frames]
        }

        # Filter the original list to get the selected frames
        selected_frames = [
            f for f in frames if f["timestamp"] in selected_timestamps
        ]

        # Return the selected frames sorted by timestamp
        return sorted(selected_frames, key=lambda x: x["timestamp"])

    def extract_segment_frames(
        self, segment: Dict, num_internal_frames: int = 3
    ) -> List[Dict]:
        """Extract representative frames from a given transcript segment.

        Extracts frames at the start and end times of the segment. Additionally,
        it attempts to extract `num_internal_frames` evenly spaced frames from
        within the segment's duration. To improve the selection of internal
        frames, it extracts slightly more frames than requested and then uses
        `_select_best_frames` based on quality assessment to pick the final set.

        Args:
            segment (Dict): A dictionary representing the transcript segment,
                containing 'start_time', 'end_time', and 'segment_id'.
            num_internal_frames (int, optional): The desired number of frames
                to extract from within the segment (excluding start and end).
                Defaults to 3.

        Returns:
            List[Dict]: A list of frame information dictionaries for the
            selected frames from the segment, sorted by timestamp. Each dictionary
            contains frame data, timestamp, type, quality score, and potentially
            a path to the saved screenshot. Returns an empty list if no frames
            could be extracted.
        """
        frames = []
        start_time = segment.get("start_time")
        end_time = segment.get("end_time")
        segment_id = segment.get("segment_id")

        if start_time is None or end_time is None or segment_id is None:
            logger.warning(
                f"Segment missing required keys (start_time, end_time, segment_id): {segment}"
            )
            return []

        duration = end_time - start_time
        if duration < 0:
            logger.warning(
                f"Segment {segment_id} has negative duration ({start_time} to {end_time}). Skipping."
            )
            return []

        # Always try to get start and end frames
        start_frame_info = self._extract_frame_at_time(
            start_time, "segment_start", segment_id
        )
        if start_frame_info:
            frames.append(start_frame_info)

        end_frame_info = self._extract_frame_at_time(
            end_time, "segment_end", segment_id
        )
        # Avoid adding duplicate frame if start and end times are identical
        if end_frame_info and (
            not start_frame_info or end_frame_info["timestamp"] != start_frame_info["timestamp"]
        ):
            frames.append(end_frame_info)

        # Extract internal frames if the segment is long enough and requested
        internal_frames_to_extract = []
        if num_internal_frames > 0 and duration > 1.0:  # Min duration for internal frames
            # Calculate interval, avoid division by zero
            num_intervals = num_internal_frames + 1
            interval = duration / num_intervals

            # Extract slightly more frames than needed for better quality selection
            # Extract up to num_internal + 2, but not more than one every 0.5s
            num_candidates = min(num_internal_frames + 2, int(duration / 0.5))

            candidate_times = [
                start_time + interval * (i + 1) for i in range(num_candidates)
            ]

            for i, time in enumerate(candidate_times):
                # Ensure time is within segment bounds (handle potential float inaccuracies)
                if start_time < time < end_time:
                    frame_info = self._extract_frame_at_time(
                        time, f"internal_{i+1}", segment_id
                    )
                    if frame_info:
                        internal_frames_to_extract.append(frame_info)

        # Select the best internal frames based on quality
        best_internal_frames = self._select_best_frames(
            internal_frames_to_extract, num_internal_frames
        )
        frames.extend(best_internal_frames)

        # Final sort by timestamp
        return sorted(frames, key=lambda x: x["timestamp"])

    def _assess_frame_quality(self, frame: np.ndarray) -> float:
        """Assess the quality of a video frame using image metrics.

        Calculates a quality score based on sharpness (Laplacian variance),
        brightness, contrast, and an estimation of motion blur (using Sobel gradients).
        These metrics are combined with weights to produce a single score between 0.0 and 1.0.

        Args:
            frame (np.ndarray): The input video frame in OpenCV's BGR format.

        Returns:
            float: A quality score between 0.0 (poor) and 1.0 (good). Returns 0.0
            if an error occurs during assessment.
        """
        try:
            if frame is None or frame.size == 0:
                logger.warning("Cannot assess quality of empty frame.")
                return 0.0

            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 1. Sharpness (Laplacian variance) - Higher is sharper
            # CV_64F to avoid overflow for high gradients
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            # Normalize: Observed values can range widely. Divide by a heuristic value.
            # Adjust 10000 based on typical video content if needed.
            sharpness = np.clip(laplacian_var / 10000.0, 0.0, 1.0)

            # 2. Brightness (Mean pixel intensity)
            brightness = np.mean(gray) / 255.0  # Normalize to 0-1

            # 3. Contrast (Standard deviation of pixel intensity)
            contrast = np.std(gray) / 128.0  # Normalize roughly to 0-1

            # 4. Motion Blur Estimation (Magnitude of Sobel gradients)
            # Less gradient magnitude might indicate blur. Use smaller ksize for sensitivity.
            sobelx_mag = np.mean(np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)))
            sobely_mag = np.mean(np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)))
            # Normalize based on potential max gradient values (heuristic)
            motion_score = np.clip((sobelx_mag + sobely_mag) / 500.0, 0.0, 1.0)

            # Combine scores with weights (adjust weights based on importance)
            # Example weights: Sharpness most important, then motion, then contrast/brightness
            quality_score = (
                0.4 * sharpness
                + 0.3 * motion_score
                + 0.15 * contrast
                + 0.15 * brightness
            )

            # Ensure the final score is within [0, 1]
            return min(max(quality_score, 0.0), 1.0)

        except cv2.error as e:
            logger.error(f"OpenCV error assessing frame quality: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Unexpected error assessing frame quality: {e}")
            return 0.0

    def _save_screenshot(
        self, frame: np.ndarray, segment_id: int, frame_type: str, timestamp: float
    ) -> Optional[str]:
        """Save a single frame as a screenshot image file.

        Creates a subdirectory for the segment if it doesn't exist and saves
        the frame using the specified format and quality settings.

        Args:
            frame (np.ndarray): The frame to save (in BGR format).
            segment_id (int): The ID of the segment this frame belongs to.
            frame_type (str): A descriptor for the frame (e.g., 'segment_start',
                'internal_1', 'segment_end').
            timestamp (float): The timestamp of the frame in the video, used
                in the filename.

        Returns:
            Optional[str]: The absolute path to the saved screenshot file, or
            None if saving failed.
        """
        try:
            # Create segment-specific directory
            segment_dir = os.path.join(self.screenshots_dir, f"segment_{segment_id}")
            os.makedirs(segment_dir, exist_ok=True)

            # Generate filename (ensure frame_type is filename-safe)
            safe_frame_type = "".join(
                c if c.isalnum() or c in ['_', '-'] else '_' for c in frame_type
            )
            base_name = f"{safe_frame_type}_{timestamp:.3f}"
            file_path = os.path.join(segment_dir, f"{base_name}.{self.save_format}")

            # Convert BGR (OpenCV default) to RGB for PIL/saving
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            # Save with appropriate format and quality/options
            if self.save_format == "jpg":
                image.save(
                    file_path,
                    "JPEG",
                    quality=self.compression_quality,
                    optimize=True,
                    progressive=True,
                )
            elif self.save_format == "png":
                image.save(file_path, "PNG", optimize=True)
            else:
                logger.warning(
                    f"Unsupported save format '{self.save_format}'. Defaulting to PNG."
                )
                file_path = os.path.join(segment_dir, f"{base_name}.png")
                image.save(file_path, "PNG", optimize=True)

            logger.debug(f"Saved screenshot: {file_path}")
            return file_path

        except cv2.error as e:
            logger.error(f"OpenCV error during screenshot saving prep: {e}")
            return None
        except IOError as e:
            logger.error(f"File I/O error saving screenshot {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error saving screenshot {file_path}: {e}")
            return None

    def _extract_frame_at_time(
        self, timestamp: float, frame_type: str, segment_id: Optional[int] = None
    ) -> Optional[Dict]:
        """Extract, assess, and potentially save a single frame at a specific time.

        Retrieves the frame from the video, assesses its quality, and if a
        `segment_id` is provided and the quality meets the threshold, saves it
        as a screenshot. Uses a cache to avoid re-processing the same timestamp.

        Args:
            timestamp (float): The video timestamp (in seconds) to extract the frame from.
            frame_type (str): A label indicating the frame's purpose (e.g.,
                'segment_start', 'internal').
            segment_id (Optional[int], optional): The ID of the segment this frame
                belongs to. Required if the frame should be considered for saving
                as a screenshot. Defaults to None.

        Returns:
            Optional[Dict]: A dictionary containing frame information:
                'timestamp' (float): The exact timestamp.
                'frame_type' (str): The provided frame type label.
                'frame' (np.ndarray): The frame data in BGR format.
                'quality_score' (float): The assessed quality score (0-1).
                'screenshot_path' (str, optional): Path if saved as screenshot.
            Returns None if the frame extraction or processing fails.
        """
        # Ensure timestamp is within video duration
        if not (0 <= timestamp <= self.video.duration):
            logger.warning(
                f"Timestamp {timestamp:.3f}s is outside video duration (0-{self.video.duration:.3f}s). Skipping."
            )
            return None

        try:
            # Format cache key consistently
            cache_key = f"{timestamp:.3f}"
            if cache_key in self.frame_cache:
                # Return cached data, potentially updating segment_id if needed
                cached_info = self.frame_cache[cache_key].copy()
                # If this call provides a segment_id and the cached one didn't have one
                # (or didn't save), re-evaluate saving.
                if (
                    segment_id is not None
                    and "screenshot_path" not in cached_info
                    and cached_info["quality_score"] >= self.quality_threshold
                ):
                    screenshot_path = self._save_screenshot(
                        cached_info["frame"], segment_id, frame_type, timestamp
                    )
                    if screenshot_path:
                        cached_info["screenshot_path"] = screenshot_path
                        # Update cache with the saved path
                        self.frame_cache[cache_key] = cached_info
                return cached_info

            # Extract frame using moviepy
            frame_rgb_moviepy = self.video.get_frame(timestamp)

            # Convert frame from RGB (moviepy) to BGR (OpenCV)
            frame_bgr = cv2.cvtColor(frame_rgb_moviepy, cv2.COLOR_RGB2BGR)

            # Assess frame quality
            quality_score = self._assess_frame_quality(frame_bgr)

            frame_info = {
                "timestamp": timestamp,
                "frame_type": frame_type,
                "frame": frame_bgr,  # Store BGR frame for consistency
                "quality_score": quality_score,
            }

            # Save screenshot if quality is sufficient and segment context is provided
            screenshot_path = None
            if segment_id is not None and quality_score >= self.quality_threshold:
                screenshot_path = self._save_screenshot(
                    frame_bgr, segment_id, frame_type, timestamp
                )
                if screenshot_path:
                    frame_info["screenshot_path"] = screenshot_path

            # Cache the result (including potential screenshot path)
            self.frame_cache[cache_key] = frame_info.copy() # Cache a copy
            # Don't cache the actual frame array if memory is a concern
            # self.frame_cache[cache_key].pop('frame', None)

            return frame_info

        except IndexError:
            # moviepy might raise IndexError for times slightly out of bounds
            logger.warning(
                f"IndexError extracting frame at {timestamp:.3f}s. Might be slightly out of bounds."
            )
            return None
        except Exception as e:
            # Catch potential errors from moviepy, cv2, or quality assessment
            logger.error(
                f"Error extracting or processing frame at {timestamp:.3f}s: {e}",
                exc_info=True, # Include traceback
            )
            return None

    def analyze_frame_with_context(
        self,
        frame_info: Dict,
        segment_context: Dict,
        previous_analysis: Optional[Dict] = None,
    ) -> Dict:
        """Analyze a single frame using visual data and contextual information.

        Performs software detection (OCR, logo) on the frame and then uses a
        multimodal model (Gemini) to generate a description, incorporating
        context from the transcript segment, topic, keywords, software detection
        results, and optionally, the analysis of the preceding frame.

        Args:
            frame_info (Dict): A dictionary containing the frame data ('frame'),
                timestamp ('timestamp'), and type ('frame_type').
            segment_context (Dict): A dictionary containing contextual information
                for the segment, such as 'transcript', 'dominant_topic', and
                'top_keywords'.
            previous_analysis (Optional[Dict], optional): The analysis result
                dictionary from the previously analyzed frame in the sequence,
                if available. Used to provide temporal context to the LLM.
                Defaults to None.

        Returns:
            Dict: A dictionary containing the analysis results for the frame:
                'timestamp' (float): Frame timestamp.
                'frame_type' (str): Frame type label.
                'software_analysis' (Dict): Results from OCR and logo detection.
                    Contains 'ocr_matches' and 'logo_matches'.
                'gemini_analysis' (str): Textual analysis generated by the
                    Gemini model. Contains error message if analysis failed.
        """
        frame = frame_info.get("frame")
        timestamp = frame_info.get("timestamp", -1.0)
        frame_type = frame_info.get("frame_type", "unknown")

        if frame is None:
            logger.warning(f"Cannot analyze frame at {timestamp}: Frame data missing.")
            return {
                "timestamp": timestamp,
                "frame_type": frame_type,
                "software_analysis": {"ocr_matches": [], "logo_matches": []},
                "gemini_analysis": "Analysis failed: Frame data missing.",
            }

        # Perform software detection (only if software list is provided)
        software_analysis = {"ocr_matches": [], "logo_matches": []}
        if self.software_list:
            try:
                software_analysis["ocr_matches"] = detect_software_names(
                    frame, self.software_list, self.ocr_lang
                )
            except Exception as e:
                logger.error(f"OCR detection failed for frame at {timestamp}: {e}")
            try:
                software_analysis["logo_matches"] = detect_software_logos(
                    frame, self.software_list, self.logo_db_path, self.logo_threshold
                )
            except Exception as e:
                logger.error(f"Logo detection failed for frame at {timestamp}: {e}")

        # Build context string for the multimodal model
        context_prompt = self._build_analysis_context(
            frame_info, segment_context, software_analysis, previous_analysis
        )

        # Prepare image for Gemini (expects PIL Image in RGB)
        try:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        except cv2.error as e:
            logger.error(f"Failed to convert frame to PIL Image at {timestamp}: {e}")
            return {
                "timestamp": timestamp,
                "frame_type": frame_type,
                "software_analysis": software_analysis,
                "gemini_analysis": f"Analysis failed: Frame conversion error ({e}).",
            }

        # Get analysis from Gemini
        gemini_analysis_text = "Analysis skipped or failed."
        try:
            # Assuming analyze_with_gemini takes prompt and PIL image
            gemini_analysis_text = analyze_with_gemini(context_prompt, pil_image)
            logger.debug(f"Gemini analysis successful for frame at {timestamp}")
        except Exception as e:
            logger.error(
                f"Gemini API call failed for frame at {timestamp}: {e}", exc_info=True
            )
            gemini_analysis_text = f"Analysis failed due to API error: {e}"

        return {
            "timestamp": timestamp,
            "frame_type": frame_type,
            "software_analysis": software_analysis,
            "gemini_analysis": gemini_analysis_text,
            # Include quality score if available in frame_info
            "quality_score": frame_info.get("quality_score"),
            # Include screenshot path if available
            "screenshot_path": frame_info.get("screenshot_path"),
        }

    def _build_analysis_context(
        self,
        frame_info: Dict,
        segment_context: Dict,
        software_analysis: Dict,
        previous_analysis: Optional[Dict],
    ) -> str:
        """Construct the textual context/prompt for multimodal frame analysis.

        Combines information about the frame itself (timestamp, type), the
        associated transcript segment (text, topic, keywords), software
        detection results, and optionally, the analysis of the previous frame
        into a single string prompt.

        Args:
            frame_info (Dict): Information about the current frame.
            segment_context (Dict): Context from the transcript segment.
            software_analysis (Dict): Results from software detection.
            previous_analysis (Optional[Dict]): Analysis results from the
                preceding frame, if available.

        Returns:
            str: A formatted string containing all the contextual information,
            suitable for use as a prompt for a multimodal LLM.
        """
        context_parts = [
            f"Analyze the following video frame.",
            f"Timestamp: {frame_info.get('timestamp', 'N/A'):.2f}s",
            f"Frame Type: {frame_info.get('frame_type', 'N/A')}",
            f"Frame Quality Score: {frame_info.get('quality_score', 'N/A'):.2f}",
        ]

        # Add segment context if available
        if segment_context:
            context_parts.append("\nAssociated Transcript Segment Context:")
            context_parts.append(f"  Transcript: {segment_context.get('transcript', 'N/A')}")
            context_parts.append(f"  Dominant Topic: {segment_context.get('dominant_topic', 'N/A')}")
            keywords = segment_context.get('top_keywords', [])
            context_parts.append(f"  Keywords: {', '.join(keywords) if keywords else 'N/A'}")

        # Add software detection context (only if detection was attempted/successful)
        if self.software_list and (
            software_analysis.get("ocr_matches") or software_analysis.get("logo_matches")
        ):
            context_parts.append("\nSoftware Detection Results:")
            ocr_matches = software_analysis.get("ocr_matches", [])
            if ocr_matches:
                matches_str = ", ".join(
                    [f"{m['software']} (found text: '{m['detected_text']}')" for m in ocr_matches]
                )
                context_parts.append(f"  Text-based Detections: {matches_str}")
            else:
                 context_parts.append(f"  Text-based Detections: None")

            logo_matches = software_analysis.get("logo_matches", [])
            if logo_matches:
                matches_str = ", ".join(
                    [f"{m['software']} (confidence: {m['confidence']:.2f})" for m in logo_matches]
                )
                context_parts.append(f"  Logo-based Detections: {matches_str}")
            else:
                context_parts.append(f"  Logo-based Detections: None")

        # Add previous frame analysis context if available
        if previous_analysis and "gemini_analysis" in previous_analysis:
            prev_ts = previous_analysis.get('timestamp', 'N/A')
            prev_analysis_text = previous_analysis['gemini_analysis']
            # Limit length of previous analysis to avoid overly long prompts
            max_prev_len = 300
            if len(prev_analysis_text) > max_prev_len:
                prev_analysis_text = prev_analysis_text[:max_prev_len] + "..."

            context_parts.append(
                f"\nContext from Previous Frame (at {prev_ts:.2f}s):"
                f"\n---\n{prev_analysis_text}\n---"
            )

        # Add instruction for the LLM
        context_parts.append(
            "\nTask: Describe the visual content of the frame, focusing on:"
            "\n- What software application or interface is visible (if any)?"
            "\n- What specific actions or elements are shown (e.g., code, menus, diagrams, buttons)? "
            "\n- How does the visual content relate to the transcript context and detected software?"
            "\n- Note any significant visual changes compared to the previous frame context (if provided)."
            "\nBe concise and informative."
        )

        return "\n".join(context_parts)

    def _get_visual_summary(self, frame_analyses: List[Dict]) -> Dict:
        """Generate a consolidated summary of visual content from multiple frame analyses.

        Aggregates information like detected software (with counts), notable
        visual elements mentioned in Gemini analyses, the total number of frames
        analyzed, and paths to any saved screenshots for the segment.

        Args:
            frame_analyses (List[Dict]): A list of analysis result dictionaries,
                one for each analyzed frame in a segment.

        Returns:
            Dict: A dictionary summarizing the visual content:
                'detected_software' (List[Dict]): List of unique software detected,
                    with names and occurrence counts.
                'visual_elements' (List[str]): List of unique notable visual
                    elements identified from Gemini analyses.
                'frame_count' (int): Total number of frames analyzed for the segment.
                'screenshot_paths' (List[str]): List of paths to saved screenshots
                    for this segment.
        """
        software_mentions = {}
        visual_elements_set = set()
        screenshot_paths = []
        processed_timestamps = set() # To avoid double counting if frame info appears multiple times

        for analysis in frame_analyses:
            timestamp = analysis.get("timestamp")
            if timestamp is None or timestamp in processed_timestamps:
                continue
            processed_timestamps.add(timestamp)

            # Aggregate software detections
            sw_analysis = analysis.get("software_analysis", {})
            for match_type in ["ocr_matches", "logo_matches"]:
                for match in sw_analysis.get(match_type, []):
                    software_name = match.get("software")
                    if software_name:
                        software_mentions[software_name] = software_mentions.get(software_name, 0) + 1

            # Extract visual elements from Gemini analysis (simple keyword check)
            gemini_text = analysis.get("gemini_analysis", "")
            if isinstance(gemini_text, str):
                # Keywords indicating potentially interesting visual elements
                keywords_to_find = [
                    "window", "button", "menu", "toolbar", "icon",
                    "chart", "graph", "diagram", "table",
                    "code", "script", "terminal", "command line",
                    "interface", "dialog box", "popup", "slider",
                    "text editor", "browser", "file explorer",
                ]
                text_lower = gemini_text.lower()
                for keyword in keywords_to_find:
                    if keyword in text_lower:
                        # Add a slightly more descriptive element if possible, else just keyword
                        # This is a basic heuristic
                        start_idx = text_lower.find(keyword)
                        sentence_end = text_lower.find('.', start_idx)
                        if sentence_end == -1: sentence_end = len(text_lower)
                        context_phrase = gemini_text[max(0, start_idx-15):min(len(gemini_text), sentence_end+1)].strip()
                        # Avoid adding overly long phrases
                        if len(context_phrase) < 100:
                           visual_elements_set.add(context_phrase)
                        else:
                           visual_elements_set.add(keyword.capitalize()) # Fallback

            # Collect screenshot paths
            if "screenshot_path" in analysis and analysis["screenshot_path"]:
                screenshot_paths.append(analysis["screenshot_path"])

        # Format detected software with counts
        detected_software_summary = [
            {"name": name, "occurrences": count}
            for name, count in sorted(
                software_mentions.items(), key=lambda item: item[1], reverse=True
            )
        ]

        return {
            "detected_software": detected_software_summary,
            "visual_elements": sorted(list(visual_elements_set)),
            "frame_count": len(processed_timestamps), # Count unique frames processed
            "screenshot_paths": sorted(list(set(screenshot_paths))), # Unique paths
        }

    def analyze_segment(self, segment: Dict) -> Dict:
        """Perform end-to-end analysis for a single transcript segment.

        Extracts representative frames for the segment, analyzes each frame
        individually using `analyze_frame_with_context`, and then generates
        a consolidated `visual_summary` for the entire segment based on the
        individual frame analyses.

        Args:
            segment (Dict): A dictionary representing the transcript segment,
                containing 'segment_id', 'start_time', 'end_time', 'transcript',
                'dominant_topic', and 'top_keywords'.

        Returns:
            Dict: A dictionary containing the comprehensive analysis for the segment:
                'segment_id' (int): The segment's ID.
                'start_time' (float): Segment start time.
                'end_time' (float): Segment end time.
                'transcript' (str): The transcript text for the segment.
                'topic' (str): The dominant topic assigned to the segment.
                'keywords' (List[str]): Top keywords for the segment.
                'visual_summary' (Dict): Aggregated visual information from frames
                    (see `_get_visual_summary`).
                'frame_analyses' (List[Dict]): Detailed analysis results for each
                    individual frame processed within the segment.
        """
        segment_id = segment.get("segment_id", "UNKNOWN")
        logger.info(f"Analyzing segment {segment_id}...")

        # 1. Extract representative frames for the segment
        # Uses quality assessment and selection internally
        frames_info = self.extract_segment_frames(segment)
        if not frames_info:
            logger.warning(f"No valid frames extracted for segment {segment_id}.")
            # Return structure with empty analysis if no frames
            return {
                "segment_id": segment_id,
                "start_time": segment.get("start_time"),
                "end_time": segment.get("end_time"),
                "transcript": segment.get("transcript"),
                "topic": segment.get("dominant_topic"),
                "keywords": segment.get("top_keywords"),
                "visual_summary": self._get_visual_summary([]), # Empty summary
                "frame_analyses": [],
            }

        logger.info(f"Extracted {len(frames_info)} frames for segment {segment_id}.")

        # 2. Analyze each selected frame with context
        frame_analyses = []
        previous_analysis = None # Context for the first frame is None

        for frame_info in frames_info:
            logger.debug(f"Analyzing frame at {frame_info['timestamp']:.2f}s for segment {segment_id}")
            analysis = self.analyze_frame_with_context(
                frame_info, segment, previous_analysis
            )
            frame_analyses.append(analysis)
            previous_analysis = analysis # Use current analysis as context for the next

        # 3. Generate the consolidated visual summary for the segment
        visual_summary = self._get_visual_summary(frame_analyses)
        logger.info(f"Generated visual summary for segment {segment_id}.")

        # 4. Combine all information
        return {
            "segment_id": segment_id,
            "start_time": segment.get("start_time"),
            "end_time": segment.get("end_time"),
            "transcript": segment.get("transcript"),
            "topic": segment.get("dominant_topic"), # Ensure key consistency
            "keywords": segment.get("top_keywords"), # Ensure key consistency
            "visual_summary": visual_summary,
            "frame_analyses": frame_analyses, # Include detailed frame analyses
        }

    def close(self):
        """Release video file resources."""
        if hasattr(self, 'video') and self.video:
            try:
                self.video.close()
                logger.info("Video file resources released.")
            except Exception as e:
                logger.error(f"Error closing video file: {e}")
        self.frame_cache.clear() # Clear cache on close
