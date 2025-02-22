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
        software_list: Optional[List[str]] = None,
        logo_db_path: Optional[str] = None,
        ocr_lang: str = "eng",
        logo_threshold: float = 0.8,
    ):
        """Initialize the analyzer.

        Args:
            video_path: Path to the video file
            transcript_segments: List of transcript segments with timestamps
            software_list: Optional list of software names to detect
            logo_db_path: Optional path to logo database directory
            ocr_lang: Language for OCR detection
            logo_threshold: Confidence threshold for logo detection
        """
        self.video = VideoFileClip(video_path)
        self.segments = transcript_segments
        self.software_list = software_list
        self.logo_db_path = logo_db_path
        self.ocr_lang = ocr_lang
        self.logo_threshold = logo_threshold
        self.frame_cache = {}  # Cache analyzed frames to avoid reprocessing

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

        # Always get start and end frames
        frames.append(self._extract_frame_at_time(start_time, "segment_start"))
        frames.append(self._extract_frame_at_time(end_time, "segment_end"))

        # Get internal frames at evenly spaced intervals
        if num_internal_frames > 0 and duration > 2:  # Only if segment is long enough
            interval = duration / (num_internal_frames + 1)
            for i in range(num_internal_frames):
                time = start_time + interval * (i + 1)
                frames.append(self._extract_frame_at_time(time, "internal"))

        # Remove any None values from failed extractions
        return [f for f in frames if f is not None]

    def _extract_frame_at_time(
        self, timestamp: float, frame_type: str
    ) -> Optional[Dict]:
        """Extract a single frame at a specific timestamp.

        Args:
            timestamp: Video timestamp in seconds
            frame_type: Type of frame (segment_start, segment_end, internal)

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

            frame_info = {
                "timestamp": timestamp,
                "frame_type": frame_type,
                "frame": frame_rgb,
            }

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
            frame_analyses.append(analysis)
            previous_analysis = analysis

        return {
            "segment_id": segment["segment_id"],
            "start_time": segment["start_time"],
            "end_time": segment["end_time"],
            "transcript": segment["transcript"],
            "topic": segment["dominant_topic"],
            "keywords": segment["top_keywords"],
            "frame_analyses": frame_analyses,
        }

    def close(self):
        """Clean up resources."""
        if self.video:
            self.video.close()
