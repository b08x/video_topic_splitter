#!/usr/bin/env python3
"""Contextual frame analysis functionality."""

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
    """Handles frame extraction and analysis with transcript context."""

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
        """Initialize the analyzer.

        Args:
            video_path: Path to the video file
            transcript_segments: List of transcript segments with timestamps
            project_path: Path to project directory for saving screenshots
            software_list: Optional list of software names to detect
            logo_db_path: Optional path to logo database directory
            ocr_lang: Language for OCR detection
            logo_threshold: Confidence threshold for logo detection
            quality_threshold: Threshold for frame quality assessment (0-1)
            save_format: Format to save screenshots (jpg/png)
            compression_quality: JPEG compression quality (1-100)
        """
        self.video = VideoFileClip(video_path)
        self.segments = transcript_segments
        self.project_path = project_path
        self.screenshots_dir = os.path.join(project_path, "screenshots")
        self.software_list = software_list
        self.logo_db_path = logo_db_path
        self.ocr_lang = ocr_lang
        self.logo_threshold = logo_threshold
        self.quality_threshold = quality_threshold
        self.save_format = save_format()
        self.compression_quality = compression_quality
        self.frame_cache = {}  # Cache analyzed frames to avoid reprocessing

        # Create screenshots directory structure
        os.makedirs(self.screenshots_dir, exist_ok=True)

    def _select_best_frames(
        self, frames: List[Dict], max_frames: int = 3
    ) -> List[Dict]:
        """Select the best frames based on quality scores.

        Args:
            frames: List of frame information dictionaries
            max_frames: Maximum number of frames to select

        Returns:
            List of best frames sorted by quality
        """
        # Sort frames by quality score
        sorted_frames = sorted(
            frames, key=lambda x: x.get("quality_score", 0), reverse=True
        )

        # Select top frames while maintaining temporal order
        if len(sorted_frames) <= max_frames:
            return sorted(sorted_frames, key=lambda x: x["timestamp"])

        # Get timestamps of highest quality frames
        selected_times = [f["timestamp"] for f in sorted_frames[:max_frames]]
        return sorted(
            [f for f in frames if f["timestamp"] in selected_times],
            key=lambda x: x["timestamp"],
        )

    def extract_segment_frames(
        self, segment: Dict, num_internal_frames: int = 3
    ) -> List[Dict]:
        """Extract frames from a segment at key points.

        Args:
            segment: Transcript segment with start/end times
            num_internal_frames: Number of frames to extract within segment

        Returns:
            List of frame information dictionaries
        """
        frames = []
        start_time = segment["start_time"]
        end_time = segment["end_time"]
        duration = end_time - start_time
        segment_id = segment["segment_id"]

        # Always get start and end frames
        frames.append(
            self._extract_frame_at_time(start_time, "segment_start", segment_id)
        )
        frames.append(self._extract_frame_at_time(end_time, "segment_end", segment_id))

        # Get internal frames at evenly spaced intervals
        if num_internal_frames > 0 and duration > 2:  # Only if segment is long enough
            interval = duration / (num_internal_frames + 1)
            # Extract more internal frames than needed for quality selection
            num_extra_frames = min(num_internal_frames + 2, int(duration / 2))

            for i in range(num_extra_frames):
                time = start_time + interval * (i + 1)
                frames.append(self._extract_frame_at_time(time, "internal", segment_id))

        # Remove any None values from failed extractions
        valid_frames = [f for f in frames if f is not None]

        # Select best frames based on quality
        return self._select_best_frames(valid_frames, num_internal_frames + 2)

    def _assess_frame_quality(self, frame: np.ndarray) -> float:
        """Assess the quality of a frame.

        Args:
            frame: OpenCV format frame (BGR)

        Returns:
            Quality score between 0 and 1
        """
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate sharpness using Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = np.var(laplacian) / 10000  # Normalize to 0-1 range

            # Calculate brightness
            brightness = np.mean(gray) / 255  # Normalize to 0-1

            # Calculate contrast
            contrast = np.std(gray) / 128  # Normalize to approximate 0-1

            # Detect motion blur
            # Use horizontal and vertical Sobel filters
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            motion_score = (np.mean(np.abs(sobelx)) + np.mean(np.abs(sobely))) / 255

            # Combine scores with weights
            quality_score = (
                0.4 * sharpness
                + 0.3 * motion_score  # Sharpness is most important
                + 0.15 * brightness  # Motion blur detection
                + 0.15 * contrast  # Brightness  # Contrast
            )

            return min(max(quality_score, 0.0), 1.0)  # Ensure between 0 and 1

        except Exception as e:
            logger.error(f"Error assessing frame quality: {str(e)}")
            return 0.0

    def _save_screenshot(
        self, frame: np.ndarray, segment_id: int, frame_type: str, timestamp: float
    ) -> Optional[str]:
        """Save a frame as a screenshot.

        Args:
            frame: OpenCV format frame (BGR)
            segment_id: ID of the segment
            frame_type: Type of frame (segment_start, segment_end, internal)
            timestamp: Frame timestamp

        Returns:
            Path to saved screenshot or None if save failed
        """
        try:
            # Create segment directory
            segment_dir = os.path.join(self.screenshots_dir, f"segment_{segment_id}")
            os.makedirs(segment_dir, exist_ok=True)

            # Generate filename
            base_name = f"{frame_type}_{timestamp:.3f}"
            file_path = os.path.join(segment_dir, f"{base_name}.{self.save_format}")

            # Convert BGR to RGB for saving
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            # Save with appropriate format and quality
            if self.save_format == "jpg":
                image.save(file_path, "JPEG", quality=self.compression_quality)
            else:  # png
                image.save(file_path, "PNG", optimize=True)

            return file_path

        except Exception as e:
            logger.error(f"Error saving screenshot: {str(e)}")
            return None

    def _extract_frame_at_time(
        self, timestamp: float, frame_type: str, segment_id: Optional[int] = None
    ) -> Optional[Dict]:
        """Extract a single frame at a specific timestamp.

        Args:
            timestamp: Video timestamp in seconds
            frame_type: Type of frame (segment_start, segment_end, internal)
            segment_id: Optional segment ID for saving screenshots

        Returns:
            Frame information dictionary or None if extraction fails
        """
        try:
            # Check if frame is already cached
            cache_key = f"{timestamp:.3f}"
            if cache_key in self.frame_cache:
                return self.frame_cache[cache_key]

            # Extract frame
            frame = self.video.get_frame(timestamp)

            # Convert to format needed for analysis
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Assess frame quality
            quality_score = self._assess_frame_quality(frame_rgb)

            frame_info = {
                "timestamp": timestamp,
                "frame_type": frame_type,
                "frame": frame_rgb,
                "quality_score": quality_score,
            }

            # Save screenshot if quality meets threshold and segment_id provided
            if segment_id is not None and quality_score >= self.quality_threshold:
                screenshot_path = self._save_screenshot(
                    frame_rgb, segment_id, frame_type, timestamp
                )
                if screenshot_path:
                    frame_info["screenshot_path"] = screenshot_path

            # Cache the frame info
            self.frame_cache[cache_key] = frame_info
            return frame_info

        except Exception as e:
            logger.error(f"Error extracting frame at {timestamp}: {str(e)}")
            return None

    def analyze_frame_with_context(
        self,
        frame_info: Dict,
        segment_context: Dict,
        previous_analysis: Optional[Dict] = None,
    ) -> Dict:
        """Analyze a frame with transcript and topic context.

        Args:
            frame_info: Frame information dictionary
            segment_context: Segment context including transcript and topic
            previous_analysis: Optional previous frame analysis results

        Returns:
            Analysis results dictionary
        """
        frame = frame_info["frame"]
        timestamp = frame_info["timestamp"]

        # Perform software detection
        software_analysis = {
            "ocr_matches": detect_software_names(
                frame, self.software_list, self.ocr_lang
            ),
            "logo_matches": detect_software_logos(
                frame, self.software_list, self.logo_db_path, self.logo_threshold
            ),
        }

        # Build context for Gemini analysis
        context = self._build_analysis_context(
            frame_info, segment_context, software_analysis, previous_analysis
        )

        # Convert frame for Gemini
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Get Gemini analysis
        try:
            gemini_analysis = analyze_with_gemini(context, pil_image)
        except Exception as e:
            logger.error(f"Gemini analysis failed: {str(e)}")
            gemini_analysis = f"Analysis failed: {str(e)}"

        return {
            "timestamp": timestamp,
            "frame_type": frame_info["frame_type"],
            "software_analysis": software_analysis,
            "gemini_analysis": gemini_analysis,
        }

    def _build_analysis_context(
        self,
        frame_info: Dict,
        segment_context: Dict,
        software_analysis: Dict,
        previous_analysis: Optional[Dict],
    ) -> str:
        """Build context string for frame analysis.

        Args:
            frame_info: Frame information
            segment_context: Segment context
            software_analysis: Software detection results
            previous_analysis: Previous frame analysis if available

        Returns:
            Context string for analysis
        """
        context_parts = [
            f"Analyzing frame at {frame_info['timestamp']:.2f}s",
            f"Frame type: {frame_info['frame_type']}",
            f"\nTranscript context: {segment_context['transcript']}",
            f"Topic: {segment_context['dominant_topic']}",
            f"Keywords: {', '.join(segment_context['top_keywords'])}",
        ]

        # Add software detection context
        if self.software_list:
            context_parts.append("\nSoftware Detection Results:")
            if software_analysis["ocr_matches"]:
                matches = [
                    f"{m['software']} ({m['detected_text']})"
                    for m in software_analysis["ocr_matches"]
                ]
                context_parts.append(f"Text detected: {', '.join(matches)}")
            if software_analysis["logo_matches"]:
                matches = [
                    f"{m['software']} (confidence: {m['confidence']:.2f})"
                    for m in software_analysis["logo_matches"]
                ]
                context_parts.append(f"Logos detected: {', '.join(matches)}")

        # Add previous analysis context if available
        if previous_analysis:
            context_parts.append(
                f"\nPrevious frame analysis ({previous_analysis['timestamp']:.2f}s):"
                f"\n{previous_analysis['gemini_analysis']}"
            )

        return "\n".join(context_parts)

    def _get_visual_summary(self, frame_analyses: List[Dict]) -> Dict:
        """Generate a summary of visual content from frame analyses.

        Args:
            frame_analyses: List of frame analysis results

        Returns:
            Dictionary containing visual content summary
        """
        # Collect all software detections
        software_mentions = []
        for analysis in frame_analyses:
            if "software_analysis" in analysis:
                for match in analysis["software_analysis"].get("ocr_matches", []):
                    software_mentions.append(match["software"])
                for match in analysis["software_analysis"].get("logo_matches", []):
                    software_mentions.append(match["software"])

        # Get unique software mentions with counts
        software_counts = {}
        for software in software_mentions:
            software_counts[software] = software_counts.get(software, 0) + 1

        # Extract key visual elements from Gemini analyses
        visual_elements = []
        for analysis in frame_analyses:
            if "gemini_analysis" in analysis and isinstance(
                analysis["gemini_analysis"], str
            ):
                # Look for UI elements, charts, diagrams, code blocks in analysis
                # Process each line of the analysis text
                for element in analysis["gemini_analysis"].split("\n"):
                    # Ensure we're working with a string
                    element = element.strip()
                    if element:  # Skip empty lines
                        element_lower = element.lower()
                        # Check for visual elements
                        if any(
                            keyword in element_lower
                            for keyword in [
                                "window",
                                "button",
                                "menu",
                                "chart",
                                "diagram",
                                "code",
                            ]
                        ):
                            visual_elements.append(element)

        return {
            "detected_software": [
                {"name": k, "occurrences": v}
                for k, v in sorted(
                    software_counts.items(), key=lambda x: x[1], reverse=True
                )
            ],
            "visual_elements": list(set(visual_elements)),  # Unique elements
            "frame_count": len(frame_analyses),
            "screenshot_paths": [
                analysis.get("screenshot_path")
                for analysis in frame_analyses
                if "screenshot_path" in analysis
            ],
        }

    def analyze_segment(self, segment: Dict) -> Dict:
        """Analyze all frames for a segment.

        Args:
            segment: Transcript segment information

        Returns:
            Segment analysis results
        """
        # Extract frames for the segment
        frames = self.extract_segment_frames(segment)

        # Analyze each frame with context
        frame_analyses = []
        previous_analysis = None

        for frame_info in frames:
            analysis = self.analyze_frame_with_context(
                frame_info, segment, previous_analysis
            )
            # Include quality score and screenshot path if available
            if "quality_score" in frame_info:
                analysis["quality_score"] = frame_info["quality_score"]
            if "screenshot_path" in frame_info:
                analysis["screenshot_path"] = frame_info["screenshot_path"]

            frame_analyses.append(analysis)
            previous_analysis = analysis

        # Generate visual content summary
        visual_summary = self._get_visual_summary(frame_analyses)

        return {
            "segment_id": segment["segment_id"],
            "start_time": segment["start_time"],
            "end_time": segment["end_time"],
            "transcript": segment["transcript"],
            "topic": segment["dominant_topic"],
            "keywords": segment["top_keywords"],
            "visual_summary": visual_summary,
            "frame_analyses": frame_analyses,
        }

    def close(self):
        """Clean up resources."""
        if self.video:
            self.video.close()
