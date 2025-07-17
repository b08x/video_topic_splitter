#!/usr/bin/env python3
"""Project structure management for organized video processing output."""

import os
import re
import shutil
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ProjectStructure:
    """Manages the organized project directory structure."""
    
    def __init__(self, project_path: str):
        """
        Initialize project structure manager.
        
        Args:
            project_path: Base project directory path
        """
        self.project_path = project_path
        self.structure = self._define_structure()
    
    def _define_structure(self) -> Dict[str, str]:
        """Define the target directory structure."""
        return {
            "input_files": os.path.join(self.project_path, "input_files"),
            "transcript": os.path.join(self.project_path, "transcript"),
            "topic_segments": os.path.join(self.project_path, "topic_segments"),
            "final_analysis": os.path.join(self.project_path, "final_analysis"),
            "audio_analysis": os.path.join(self.project_path, "audio_analysis"),
            "temp": os.path.join(self.project_path, "temp")
        }
    
    def create_base_structure(self) -> None:
        """Create the base directory structure."""
        for dir_name, dir_path in self.structure.items():
            os.makedirs(dir_path, exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")
    
    def get_input_files_dir(self) -> str:
        """Get the input files directory path."""
        return self.structure["input_files"]
    
    def get_transcript_dir(self) -> str:
        """Get the transcript directory path."""
        return self.structure["transcript"]
    
    def get_topic_segments_dir(self) -> str:
        """Get the topic segments directory path."""
        return self.structure["topic_segments"]
    
    def get_final_analysis_dir(self) -> str:
        """Get the final analysis directory path."""
        return self.structure["final_analysis"]
    
    def get_temp_dir(self) -> str:
        """Get the temporary directory path."""
        return self.structure["temp"]
    
    def create_segment_directory(self, segment_num: int, topic_name: str) -> str:
        """
        Create a directory for a specific topic segment.
        
        Args:
            segment_num: Segment number (1-based)
            topic_name: Topic name from topic modeling
            
        Returns:
            Path to the created segment directory
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
        Clean topic name to be filesystem-safe.
        
        Args:
            topic_name: Raw topic name from topic modeling
            
        Returns:
            Cleaned topic name safe for filesystem
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
        Get all file paths for a specific segment.
        
        Args:
            segment_num: Segment number (1-based)
            topic_name: Topic name from topic modeling
            
        Returns:
            Dictionary of file paths for the segment
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
        Move or copy input files to the input_files directory.
        
        Args:
            video_path: Path to original video file
            transcript_path: Path to original transcript file (optional)
            
        Returns:
            Dictionary of new file paths
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
        Save transcript files to the transcript directory.
        
        Args:
            transcript_data: Raw transcript data
            processed_transcript: Processed transcript with topic information
            
        Returns:
            Dictionary of saved file paths
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
        Save final analysis to the final_analysis directory.
        
        Args:
            analysis_data: Final analysis results
            
        Returns:
            Path to saved timeline file
        """
        final_dir = self.get_final_analysis_dir()
        timeline_path = os.path.join(final_dir, "topic_timeline.json")
        
        with open(timeline_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved final analysis to: {timeline_path}")
        return timeline_path
    
    def cleanup_temp_files(self) -> None:
        """Clean up temporary files."""
        temp_dir = self.get_temp_dir()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info("Cleaned up temporary files")
    
    def get_legacy_files(self) -> List[str]:
        """
        Get list of legacy files that should be migrated or cleaned up.
        
        Returns:
            List of legacy file paths
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
        """Migrate legacy files to new structure."""
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
    Sanitize filename to be filesystem-safe.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
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