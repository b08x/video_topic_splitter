#!/usr/bin/env python3
"""
Segment-level analysis functionality.
Orchestrates multimodal analysis for individual video segments.
"""

import json
import logging
import os
from typing import Dict, List, Any, Optional

from .multimodal_analysis import MultimodalAnalyzer
from ..progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)


class SegmentProcessor:
    """Processes individual video segments with multimodal analysis."""
    
    def __init__(self, progress_tracker: ProgressTracker = None):
        """
        Initialize the segment processor.
        
        Args:
            progress_tracker: Optional progress tracker
        """
        self.progress_tracker = progress_tracker
        self.multimodal_analyzer = MultimodalAnalyzer(progress_tracker)
    
    def process_segments(
        self,
        video_path: str,
        segmented_files: List[Dict[str, Any]],
        transcript_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process all video segments with multimodal analysis.
        
        Args:
            video_path: Path to the original video file
            segmented_files: List of segment information from video segmentation
            transcript_data: Full transcript data
            
        Returns:
            List of processed segment analyses
        """
        logger.info(f"Processing {len(segmented_files)} segments")
        
        if self.progress_tracker:
            self.progress_tracker.start_phase("Segment Analysis")
        
        processed_segments = []
        total_segments = len(segmented_files)
        
        for i, segment_info in enumerate(segmented_files):
            segment_num = segment_info["segment_number"]
            topic_name = segment_info["topic"]
            
            if self.progress_tracker:
                progress = (i / total_segments) * 100
                self.progress_tracker.update_phase_progress(
                    progress, f"Processing segment {segment_num}: {topic_name}"
                )
            
            # Extract transcript for this segment
            transcript_segment = self._extract_transcript_for_segment(
                transcript_data, segment_info
            )
            
            # Perform multimodal analysis
            try:
                analysis_results = self.multimodal_analyzer.analyze_segment(
                    video_path, segment_info, transcript_segment
                )
                
                # Add segment processing metadata
                analysis_results["processing_info"] = {
                    "processed_successfully": True,
                    "transcript_segments_count": len(transcript_segment),
                    "files_validated": self._validate_segment_files(segment_info)
                }
                
                processed_segments.append(analysis_results)
                
            except Exception as e:
                logger.error(f"Error processing segment {segment_num}: {e}")
                
                # Create error result
                error_result = {
                    "segment_info": {
                        "segment_number": segment_num,
                        "topic": topic_name,
                        "error": str(e)
                    },
                    "processing_info": {
                        "processed_successfully": False,
                        "error": str(e)
                    }
                }
                processed_segments.append(error_result)
        
        if self.progress_tracker:
            self.progress_tracker.complete_phase("Segment Analysis")
        
        logger.info(f"Completed processing {len(processed_segments)} segments")
        return processed_segments
    
    def _extract_transcript_for_segment(
        self, 
        transcript_data: List[Dict[str, Any]], 
        segment_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract transcript data for a specific segment based on timing.
        
        Args:
            transcript_data: Full transcript data
            segment_info: Segment information with timing
            
        Returns:
            Transcript segments for this time range
        """
        try:
            start_time = segment_info["start_time"]
            end_time = segment_info["end_time"]
            
            segment_transcript = []
            
            for transcript_item in transcript_data:
                # Get timing information (support multiple formats)
                item_start = transcript_item.get("start", transcript_item.get("start_time", 0))
                item_end = transcript_item.get("end", transcript_item.get("end_time", 0))
                
                # Check if transcript item overlaps with segment
                if (item_start < end_time and item_end > start_time):
                    # Adjust timing to be relative to segment start
                    adjusted_item = transcript_item.copy()
                    adjusted_item["segment_start"] = max(0, item_start - start_time)
                    adjusted_item["segment_end"] = min(end_time - start_time, item_end - start_time)
                    adjusted_item["original_start"] = item_start
                    adjusted_item["original_end"] = item_end
                    
                    segment_transcript.append(adjusted_item)
            
            logger.debug(f"Extracted {len(segment_transcript)} transcript items for segment")
            return segment_transcript
            
        except Exception as e:
            logger.error(f"Error extracting transcript for segment: {e}")
            return []
    
    def _validate_segment_files(self, segment_info: Dict[str, Any]) -> Dict[str, bool]:
        """
        Validate that segment files exist and are valid.
        
        Args:
            segment_info: Segment information
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            "video_exists": False,
            "audio_exists": False,
            "frames_dir_exists": False,
            "all_files_valid": False
        }
        
        try:
            paths = segment_info.get("paths", {})
            
            # Check video file
            video_path = paths.get("video_file")
            if video_path and os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                validation["video_exists"] = True
            
            # Check audio file
            audio_path = paths.get("audio_file")
            if audio_path and os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                validation["audio_exists"] = True
            
            # Check frames directory
            frames_dir = paths.get("frames_dir")
            if frames_dir and os.path.exists(frames_dir) and os.path.isdir(frames_dir):
                validation["frames_dir_exists"] = True
            
            # Overall validation
            validation["all_files_valid"] = (
                validation["video_exists"] and 
                validation["audio_exists"] and 
                validation["frames_dir_exists"]
            )
            
        except Exception as e:
            logger.error(f"Error validating segment files: {e}")
        
        return validation
    
    def generate_segment_timeline(
        self, 
        processed_segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate a timeline of all processed segments.
        
        Args:
            processed_segments: List of processed segment analyses
            
        Returns:
            Timeline data structure
        """
        try:
            timeline = {
                "total_segments": len(processed_segments),
                "total_duration": 0,
                "segments": [],
                "topic_summary": {},
                "technical_elements": set(),
                "generation_timestamp": None
            }
            
            import time
            timeline["generation_timestamp"] = time.time()
            
            for segment in processed_segments:
                segment_info = segment.get("segment_info", {})
                multimodal_summary = segment.get("multimodal_summary", {})
                
                # Extract segment timeline info
                segment_timeline = {
                    "segment_number": segment_info.get("segment_number"),
                    "topic": segment_info.get("topic"),
                    "start_time": segment_info.get("start_time"),
                    "end_time": segment_info.get("end_time"),
                    "duration": segment_info.get("duration"),
                    "key_insights": multimodal_summary.get("key_insights", []),
                    "technical_elements": multimodal_summary.get("technical_elements", []),
                    "confidence_score": multimodal_summary.get("confidence_score", 0.0),
                    "modalities_analyzed": multimodal_summary.get("modalities_analyzed", [])
                }
                
                timeline["segments"].append(segment_timeline)
                
                # Update totals
                timeline["total_duration"] += segment_info.get("duration", 0)
                
                # Update topic summary
                topic = segment_info.get("topic", "Unknown")
                if topic not in timeline["topic_summary"]:
                    timeline["topic_summary"][topic] = {
                        "segments": [],
                        "total_duration": 0,
                        "technical_elements": set()
                    }
                
                timeline["topic_summary"][topic]["segments"].append(segment_info.get("segment_number"))
                timeline["topic_summary"][topic]["total_duration"] += segment_info.get("duration", 0)
                timeline["topic_summary"][topic]["technical_elements"].update(
                    multimodal_summary.get("technical_elements", [])
                )
                
                # Update overall technical elements
                timeline["technical_elements"].update(
                    multimodal_summary.get("technical_elements", [])
                )
            
            # Convert sets to lists for JSON serialization
            timeline["technical_elements"] = list(timeline["technical_elements"])
            for topic_data in timeline["topic_summary"].values():
                topic_data["technical_elements"] = list(topic_data["technical_elements"])
            
            return timeline
            
        except Exception as e:
            logger.error(f"Error generating segment timeline: {e}")
            return {"error": str(e)}
    
    def save_segment_results(
        self, 
        processed_segments: List[Dict[str, Any]], 
        output_structure
    ) -> Dict[str, str]:
        """
        Save all segment processing results.
        
        Args:
            processed_segments: List of processed segment analyses
            output_structure: ProjectStructure instance
            
        Returns:
            Dictionary of saved file paths
        """
        try:
            saved_files = {}
            
            # Generate and save timeline
            timeline = self.generate_segment_timeline(processed_segments)
            timeline_path = output_structure.save_final_analysis(timeline)
            saved_files["timeline"] = timeline_path
            
            # Save consolidated results
            results_path = os.path.join(
                output_structure.project_path, 
                "transcript_analysis_results.json"
            )
            
            consolidated_results = {
                "processing_summary": {
                    "total_segments": len(processed_segments),
                    "successful_segments": len([
                        s for s in processed_segments 
                        if s.get("processing_info", {}).get("processed_successfully", False)
                    ]),
                    "failed_segments": len([
                        s for s in processed_segments 
                        if not s.get("processing_info", {}).get("processed_successfully", False)
                    ])
                },
                "timeline": timeline,
                "detailed_results": processed_segments
            }
            
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(consolidated_results, f, indent=2, ensure_ascii=False, default=str)
            
            saved_files["consolidated_results"] = results_path
            
            logger.info(f"Saved segment results to {len(saved_files)} files")
            return saved_files
            
        except Exception as e:
            logger.error(f"Error saving segment results: {e}")
            return {"error": str(e)}


def filter_transcript_by_time_range(
    transcript: List[Dict[str, Any]], 
    start_time: float, 
    end_time: float
) -> List[Dict[str, Any]]:
    """
    Filter transcript to only include segments within a time range.
    
    Args:
        transcript: Full transcript data
        start_time: Start time in seconds
        end_time: End time in seconds
        
    Returns:
        Filtered transcript segments
    """
    filtered_segments = []
    
    for segment in transcript:
        # Get timing information (support multiple formats)
        seg_start = segment.get("start", segment.get("start_time", 0))
        seg_end = segment.get("end", segment.get("end_time", 0))
        
        # Check if segment overlaps with time range
        if seg_start < end_time and seg_end > start_time:
            filtered_segments.append(segment)
    
    return filtered_segments