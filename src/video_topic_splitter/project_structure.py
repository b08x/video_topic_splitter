#!/usr/bin/env python3
"""
Project structure management for organized video processing output.

This module provides the `ProjectStructure` class, which is responsible for
creating and managing a standardized directory layout for all files generated
during the video analysis process. This ensures that outputs are organized,
predictable, and easy to navigate.
"""

import os
import re
import shutil
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ProjectStructure:
    """
    Manages the organized project directory structure.

    This class defines and creates a standard folder hierarchy for each
    processing project. It provides methods to access specific directories,
    manage input and output files, and handle cleanup of temporary or
    legacy files.

    Attributes:
        project_path (str): The root path of the project directory.
        structure (Dict[str, str]): A dictionary mapping key names to their
            corresponding absolute directory paths.
    """
    
    def __init__(self, project_path: str):
        """
        Initialize the project structure manager.
        
        Args:
            project_path: The base path for the project directory.
        """
        self.project_path = project_path
        self.structure = self._define_structure()
    
    def _define_structure(self) -> Dict[str, str]:
        """
        Define the target directory structure.

        Returns:
            A dictionary defining the key subdirectories of the project.
        """
        return {
            "input_files": os.path.join(self.project_path, "input_files"),
            "transcript": os.path.join(self.project_path, "transcript"),
            "topic_segments": os.path.join(self.project_path, "topic_segments"),
            "final_analysis": os.path.join(self.project_path, "final_analysis"),
            "audio_analysis": os.path.join(self.project_path, "audio_analysis"),
            "temp": os.path.join(self.project_path, "temp")
        }
    
    def create_base_structure(self) -> None:
        """Create the base directory structure on the filesystem."""
        for dir_name, dir_path in self.structure.items():
            os.makedirs(dir_path, exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")
    
    def get_input_files_dir(self) -> str:
        """Get the path to the input files directory."""
        return self.structure["input_files"]
    
    def get_transcript_dir(self) -> str:
        """Get the path to the transcript directory."""
        return self.structure["transcript"]
    
    def get_topic_segments_dir(self) -> str:
        """Get the path to the topic segments directory."""
        return self.structure["topic_segments"]
    
    def get_final_analysis_dir(self) -> str:
        """Get the path to the final analysis directory."""
        return self.structure["final_analysis"]
    
    def get_temp_dir(self) -> str:
        """Get the path to the temporary files directory."""
        return self.structure["temp"]
    
    def create_segment_directory(self, segment_num: int, topic_name: str) -> str:
        """
        Create a dedicated directory for a specific topic segment.

        The directory name is constructed from the segment number and a
        sanitized version of the topic name. It also creates subdirectories
        for frames and audio analysis within the segment folder.
        
        Args:
            segment_num: The sequence number of the segment (1-based).
            topic_name: The name of the topic associated with the segment.
            
        Returns:
            The path to the newly created segment directory.
        """
        # Clean topic name for filesystem
        clean_topic = self._clean_topic_name(topic_name)
        segment_dir_name = f"segment_{segment_num:03d}_{clean_topic}"
        segment_dir = os.path.join(self.get_topic_segments_dir(), segment_dir_name)
        
        # Create segment directory structure
        os.makedirs(segment_dir, exist_ok=True)
        os.makedirs(os.path.join(segment_dir, "segment_frames"), exist_ok=True)
        os.makedirs(os.path.join(segment_dir, "audio_analysis"), exist_ok=True)
        
        logger.info(f"Created segment directory: {segment_dir}")
        return segment_dir
    
    def _clean_topic_name(self, topic_name: str) -> str:
        """
        Sanitize a topic name to be safe for use in a filesystem path.
        
        Args:
            topic_name: The raw topic name from topic modeling.
            
        Returns:
            A cleaned, filesystem-safe version of the topic name.
        """
        # Remove or replace problematic characters
        clean_name = re.sub(r'[<>:"/\\|?*]', '', topic_name)
        clean_name = re.sub(r'\s+', ' ', clean_name.strip())
        clean_name = clean_name.replace(' ', '_')
        
        # Limit length
        max_length = 50
        if len(clean_name) > max_length:
            clean_name = clean_name[:max_length].rstrip('_')
        
        # Ensure it's not empty
        if not clean_name:
            clean_name = "Unknown_Topic"
        
        return clean_name
    
    def get_segment_paths(self, segment_num: int, topic_name: str) -> Dict[str, str]:
        """
        Get a dictionary of all standard file paths for a specific segment.
        
        Args:
            segment_num: The sequence number of the segment (1-based).
            topic_name: The name of the topic associated with the segment.
            
        Returns:
            A dictionary mapping key file types to their full paths for the
            given segment.
        """
        segment_dir = self.create_segment_directory(segment_num, topic_name)
        
        return {
            "segment_dir": segment_dir,
            "video_file": os.path.join(segment_dir, f"segment_{segment_num:03d}.mp4"),
            "audio_file": os.path.join(segment_dir, f"segment_{segment_num:03d}_audio.wav"),
            "frames_dir": os.path.join(segment_dir, "segment_frames"),
            "audio_analysis_dir": os.path.join(segment_dir, "audio_analysis"),
            "multimodal_analysis": os.path.join(segment_dir, f"multimodal_analysis_{segment_num:03d}.json"),
            "speaker_transcript": os.path.join(segment_dir, f"speaker_attributed_transcript_{segment_num:03d}.json"),
            "segment_summary": os.path.join(segment_dir, "segment_summary.json")
        }
    
    def move_input_files(self, video_path: str, transcript_path: Optional[str] = None) -> Dict[str, str]:
        """
        Copy input files into the project's `input_files` directory.
        
        Args:
            video_path: The path to the original input video file.
            transcript_path: The optional path to the original transcript file.
            
        Returns:
            A dictionary containing the new paths of the copied files.
        """
        input_dir = self.get_input_files_dir()
        
        # Copy video file
        video_filename = os.path.basename(video_path)
        new_video_path = os.path.join(input_dir, video_filename)
        if not os.path.exists(new_video_path):
            shutil.copy2(video_path, new_video_path)
            logger.info(f"Copied video file to: {new_video_path}")
        
        result = {"video": new_video_path}
        
        # Copy transcript file if provided
        if transcript_path and os.path.exists(transcript_path):
            transcript_filename = os.path.basename(transcript_path)
            new_transcript_path = os.path.join(input_dir, transcript_filename)
            if not os.path.exists(new_transcript_path):
                shutil.copy2(transcript_path, new_transcript_path)
                logger.info(f"Copied transcript file to: {new_transcript_path}")
            result["transcript"] = new_transcript_path
        
        return result
    
    def save_transcript_files(self, transcript_data: List[Dict], processed_transcript: Dict) -> Dict[str, str]:
        """
        Save processed transcript and analysis files to the transcript directory.
        
        Args:
            transcript_data: The raw transcript data (list of segments).
            processed_transcript: The transcript after topic modeling and
                additional processing.
            
        Returns:
            A dictionary of the saved file paths.
        """
        transcript_dir = self.get_transcript_dir()
        
        # Save processed transcript
        processed_path = os.path.join(transcript_dir, "processed_transcript.json")
        with open(processed_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(processed_transcript, f, indent=2, ensure_ascii=False)
        
        # Save transcript analysis
        analysis_path = os.path.join(transcript_dir, "transcript_analysis.json")
        analysis_data = {
            "total_segments": len(transcript_data),
            "total_duration": max(seg.get("end", 0) for seg in transcript_data) if transcript_data else 0,
            "topics_identified": len(processed_transcript.get("topics", [])),
            "segments_created": len(processed_transcript.get("segments", []))
        }
        with open(analysis_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved transcript files to: {transcript_dir}")
        return {
            "processed_transcript": processed_path,
            "transcript_analysis": analysis_path
        }
    
    def save_final_analysis(self, analysis_data: Dict) -> str:
        """
        Save the final, consolidated analysis results to a JSON file.
        
        Args:
            analysis_data: The final analysis results dictionary.
            
        Returns:
            The path to the saved analysis file.
        """
        final_dir = self.get_final_analysis_dir()
        timeline_path = os.path.join(final_dir, "topic_timeline.json")
        
        with open(timeline_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved final analysis to: {timeline_path}")
        return timeline_path
    
    def cleanup_temp_files(self) -> None:
        """Remove the temporary files directory and its contents."""
        temp_dir = self.get_temp_dir()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info("Cleaned up temporary files")
    
    def get_legacy_files(self) -> List[str]:
        """
        Identify files from older versions of the tool that should be migrated.
        
        Returns:
            A list of absolute paths to legacy files found in the project root.
        """
        legacy_patterns = [
            "audio.opus",
            "audio_processed.aac", 
            "audio_normalized.aac",
            "unsilenced_video.mp4",
            "transcript.srt",
            "transcript.vtt",
            "results.json",
            "scenes/*.jpg",
            "scenes/frames/*.jpg"
        ]
        
        legacy_files = []
        for pattern in legacy_patterns:
            if '*' in pattern:
                import glob
                legacy_files.extend(glob.glob(os.path.join(self.project_path, pattern)))
            else:
                file_path = os.path.join(self.project_path, pattern)
                if os.path.exists(file_path):
                    legacy_files.append(file_path)
        
        return legacy_files
    
    def migrate_legacy_files(self) -> None:
        """Move legacy files to the temporary directory for eventual cleanup."""
        legacy_files = self.get_legacy_files()
        
        for file_path in legacy_files:
            try:
                # Move legacy files to temp directory for cleanup
                temp_dir = self.get_temp_dir()
                filename = os.path.basename(file_path)
                temp_path = os.path.join(temp_dir, filename)
                
                if os.path.exists(file_path):
                    shutil.move(file_path, temp_path)
                    logger.debug(f"Moved legacy file: {file_path} -> {temp_path}")
                    
            except Exception as e:
                logger.warning(f"Failed to migrate legacy file {file_path}: {e}")


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a string to be a filesystem-safe filename.

    Removes or replaces characters that are invalid in many filesystems,
    replaces whitespace, and truncates the name to a reasonable length.
    
    Args:
        filename: The original, potentially unsafe filename.
        
    Returns:
        A sanitized, filesystem-safe filename.
    """
    # Remove or replace problematic characters
    clean_name = re.sub(r'[<>:"/\\|?*]', '', filename)
    clean_name = re.sub(r'\s+', ' ', clean_name.strip())
    
    # Limit length
    max_length = 255
    if len(clean_name) > max_length:
        name, ext = os.path.splitext(clean_name)
        clean_name = name[:max_length-len(ext)] + ext
    
    return clean_name if clean_name else "untitled"
