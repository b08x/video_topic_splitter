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
        
        This method supports resuming from previously completed segments by checking
        for existing segment_summary.json files. If a segment was already processed,
        it will be loaded from disk instead of being reprocessed.
        
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
        
        # Check for existing completed segments
        completed_segments = self._load_completed_segments(segmented_files)
        processed_segments = []
        total_segments = len(segmented_files)
        
        # Add completed segments to results
        for segment_info in segmented_files:
            segment_num = segment_info["segment_number"]
            if segment_num in completed_segments:
                processed_segments.append(completed_segments[segment_num])
                logger.info(f"Loaded existing analysis for segment {segment_num}")
        
        # Process remaining segments
        segments_to_process = [s for s in segmented_files if s["segment_number"] not in completed_segments]
        
        if segments_to_process:
            logger.info(f"Processing {len(segments_to_process)} new segments (resuming from {len(completed_segments)} completed)")
        else:
            logger.info("All segments already completed, loading existing results")
        
        for i, segment_info in enumerate(segments_to_process):
            segment_num = segment_info["segment_number"]
            topic_name = segment_info["topic"]
            
            # Calculate progress including already completed segments
            total_processed = len(completed_segments) + i
            progress = (total_processed / total_segments) * 100
            
            if self.progress_tracker:
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
                
                # Save checkpoint after each successful segment
                self._save_segment_checkpoint(segment_num, len(completed_segments) + i + 1, total_segments)
                
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
        
        # Sort by segment number to maintain order
        processed_segments.sort(key=lambda x: x.get("segment_info", {}).get("segment_number", 0))
        
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
        Extract the relevant transcript data for a specific video segment.

        This method filters the full transcript to include only the items that
        overlap with the given segment's time range. It also adjusts the
        timing of the transcript items to be relative to the start of the
        segment.

        Args:
            transcript_data: The full transcript data for the video.
            segment_info: A dictionary containing the start and end times of
                the segment.

        Returns:
            A list of transcript items that fall within the segment's
            time range, with adjusted timing.
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
        Validate that the generated files for a segment exist and are not empty.

        This checks for the existence of the video segment file, audio file,
        and the directory for frames.

        Args:
            segment_info: A dictionary containing the paths to the segment's
                files.

        Returns:
            A dictionary with boolean flags indicating the validity of each
            file.
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
    
    def _load_completed_segments(self, segmented_files: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        Load analysis results from previously completed segments.
        
        This method scans the segment directories for existing segment_summary.json files
        and loads their analysis results. This enables resuming processing from where
        it left off if interrupted.
        
        Args:
            segmented_files: List of segment information from video segmentation
            
        Returns:
            Dictionary mapping segment numbers to their completed analysis results
        """
        completed_segments = {}
        
        for segment_info in segmented_files:
            segment_num = segment_info["segment_number"]
            
            # Check if this segment has been completed
            if self._is_segment_completed(segment_info):
                try:
                    # Load the completed analysis results
                    analysis_result = self._load_segment_analysis(segment_info)
                    if analysis_result:
                        completed_segments[segment_num] = analysis_result
                        logger.debug(f"Loaded completed segment {segment_num}")
                except Exception as e:
                    logger.warning(f"Could not load completed segment {segment_num}: {e}")
        
        return completed_segments
    
    def _is_segment_completed(self, segment_info: Dict[str, Any]) -> bool:
        """
        Check if a segment has been completed by looking for its summary file.
        
        Args:
            segment_info: Information about the segment to check
            
        Returns:
            True if the segment appears to have been completed, False otherwise
        """
        try:
            # Get segment number to construct the expected directory path
            segment_num = segment_info.get("segment_number")
            if not segment_num:
                return False
            
            # Try to get segment directory from paths first
            paths = segment_info.get("paths", {})
            segment_dir = paths.get("segment_dir")
            
            # If no segment_dir in paths, construct it from segment number
            if not segment_dir:
                # This handles cases where we're checking before paths are created
                from ..project_structure import ProjectStructure
                # We need to get the project path somehow - try to infer from progress tracker
                if hasattr(self, 'progress_tracker') and self.progress_tracker:
                    project_path = getattr(self.progress_tracker, 'project_path', None)
                    if project_path:
                        project_structure = ProjectStructure(project_path)
                        segment_dir = os.path.join(project_structure.get_topic_segments_dir(), f"segment_{segment_num:03d}")
            
            if not segment_dir or not os.path.exists(segment_dir):
                return False
            
            # Check for segment summary file (our completion marker)
            summary_path = os.path.join(segment_dir, "segment_summary.json")
            if not os.path.exists(summary_path):
                return False
            
            # Validate that the summary file is not empty and contains valid JSON
            try:
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)
                    # Basic validation - should have key fields
                    return (
                        "segment_number" in summary_data and 
                        "topic" in summary_data and
                        "duration" in summary_data
                    )
            except (json.JSONDecodeError, KeyError):
                return False
            
        except Exception as e:
            logger.debug(f"Error checking segment completion: {e}")
            return False
    
    def _load_segment_analysis(self, segment_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Load the full analysis results for a completed segment.
        
        Args:
            segment_info: Information about the segment to load
            
        Returns:
            The complete analysis results for the segment, or None if loading failed
        """
        try:
            # Get segment number to construct the expected directory path
            segment_num = segment_info.get("segment_number")
            if not segment_num:
                return None
            
            # Try to get segment directory from paths first
            paths = segment_info.get("paths", {})
            segment_dir = paths.get("segment_dir")
            
            # If no segment_dir in paths, construct it from segment number
            if not segment_dir:
                from ..project_structure import ProjectStructure
                if hasattr(self, 'progress_tracker') and self.progress_tracker:
                    project_path = getattr(self.progress_tracker, 'project_path', None)
                    if project_path:
                        project_structure = ProjectStructure(project_path)
                        segment_dir = os.path.join(project_structure.get_topic_segments_dir(), f"segment_{segment_num:03d}")
            
            if not segment_dir or not os.path.exists(segment_dir):
                return None
            
            # Load the main analysis file
            analysis_path = os.path.join(segment_dir, f"multimodal_analysis_{segment_num:03d}.json")
            if os.path.exists(analysis_path):
                with open(analysis_path, 'r', encoding='utf-8') as f:
                    analysis_data = json.load(f)
                    
                    # Add processing info to indicate this was loaded from disk
                    analysis_data["processing_info"] = {
                        "processed_successfully": True,
                        "loaded_from_checkpoint": True,
                        "files_validated": self._validate_segment_files(segment_info)
                    }
                    
                    return analysis_data
            
            # Fallback: construct analysis from summary if main file doesn't exist
            summary_path = os.path.join(segment_dir, "segment_summary.json")
            if os.path.exists(summary_path):
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)
                    
                    # Create a minimal analysis result from the summary
                    analysis_result = {
                        "segment_info": {
                            "segment_number": summary_data.get("segment_number"),
                            "topic": summary_data.get("topic"),
                            "duration": summary_data.get("duration", 0),
                            "start_time": segment_info.get("start_time", 0),
                            "end_time": segment_info.get("end_time", 0)
                        },
                        "multimodal_summary": {
                            "key_insights": summary_data.get("key_insights", []),
                            "technical_elements": summary_data.get("technical_elements", []),
                            "confidence_score": summary_data.get("confidence_score", 0.0),
                            "modalities_analyzed": summary_data.get("modalities_analyzed", [])
                        },
                        "processing_info": {
                            "processed_successfully": True,
                            "loaded_from_checkpoint": True,
                            "reconstructed_from_summary": True
                        }
                    }
                    
                    return analysis_result
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading segment analysis: {e}")
            return None
    
    def _save_segment_checkpoint(self, segment_num: int, completed_count: int, total_count: int) -> None:
        """
        Save a checkpoint after completing a segment.
        
        Args:
            segment_num: The segment number that was just completed
            completed_count: Total number of segments completed so far
            total_count: Total number of segments to process
        """
        try:
            from ..project import save_checkpoint
            from ..constants import CHECKPOINTS
            
            # Get project path from progress tracker if available
            project_path = None
            if hasattr(self, 'progress_tracker') and self.progress_tracker:
                project_path = getattr(self.progress_tracker, 'project_path', None)
            
            if project_path:
                checkpoint_data = {
                    "completed_segments": completed_count,
                    "total_segments": total_count,
                    "last_completed_segment": segment_num,
                    "progress_percentage": (completed_count / total_count) * 100
                }
                
                save_checkpoint(
                    project_path,
                    CHECKPOINTS["SEGMENTS_ANALYSIS_PROGRESS"],
                    checkpoint_data
                )
                
                logger.debug(f"Saved checkpoint: {completed_count}/{total_count} segments completed")
        
        except Exception as e:
            logger.warning(f"Could not save segment checkpoint: {e}")
    
    def generate_segment_timeline(
        self, 
        processed_segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate a structured timeline of all processed video segments.

        This method compiles information from all processed segments into a
        single timeline object that summarizes the topics, durations, key
        insights, and technical elements for the entire video.

        Args:
            processed_segments: A list of analysis results for each segment.

        Returns:
            A dictionary representing the complete timeline of the video,
            with summaries by topic.
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
        Save all segment processing results to the project directory.

        This includes saving the generated timeline and a consolidated JSON
        file with all detailed analysis results.

        Args:
            processed_segments: A list of analysis results for each segment.
            output_structure: An instance of ProjectStructure to manage file
                paths.

        Returns:
            A dictionary of paths to the saved files.
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