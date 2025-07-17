#!/usr/bin/env python3
"""
Simplified multimodal analysis module for topic segments.
Combines audio, visual, and transcript analysis for comprehensive insights.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
import time

from ..api.gemini import analyze_with_gemini
from ..processing.video.video_segmentation import extract_segment_frames
from ..progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)


class MultimodalAnalyzer:
    """Simplified multimodal analyzer for topic segments."""
    
    def __init__(self, progress_tracker: ProgressTracker = None):
        """
        Initialize the multimodal analyzer.
        
        Args:
            progress_tracker: Optional progress tracker
        """
        self.progress_tracker = progress_tracker
    
    def analyze_segment(
        self,
        video_path: str,
        segment_info: Dict[str, Any],
        transcript_segment: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive multimodal analysis of a video segment.
        
        Args:
            video_path: Path to the original video file
            segment_info: Segment information from video segmentation
            transcript_segment: Transcript data for this segment
            
        Returns:
            Comprehensive analysis results
        """
        segment_num = segment_info["segment_number"]
        topic_name = segment_info["topic"]
        paths = segment_info["paths"]
        
        logger.info(f"Starting multimodal analysis for segment {segment_num}: {topic_name}")
        
        if self.progress_tracker:
            self.progress_tracker.update_phase_progress(
                0.0, f"Analyzing segment {segment_num}: {topic_name}"
            )
        
        analysis_results = {
            "segment_info": {
                "segment_number": segment_num,
                "topic": topic_name,
                "start_time": segment_info["start_time"],
                "end_time": segment_info["end_time"],
                "duration": segment_info["duration"]
            },
            "transcript_analysis": self._analyze_transcript(transcript_segment),
            "visual_analysis": self._analyze_visual_content(
                video_path, 
                paths["frames_dir"], 
                segment_info["start_time"], 
                segment_info["end_time"]
            ),
            "audio_analysis": self._analyze_audio_content(paths["audio_file"]),
            "multimodal_summary": {}
        }
        
        # Create comprehensive multimodal summary
        analysis_results["multimodal_summary"] = self._create_multimodal_summary(
            analysis_results["transcript_analysis"],
            analysis_results["visual_analysis"],
            analysis_results["audio_analysis"]
        )
        
        # Save analysis results
        self._save_analysis_results(paths, analysis_results, segment_num)
        
        if self.progress_tracker:
            self.progress_tracker.update_phase_progress(
                100.0, f"Completed analysis for segment {segment_num}"
            )
        
        logger.info(f"Completed multimodal analysis for segment {segment_num}")
        return analysis_results
    
    def _analyze_transcript(self, transcript_segment: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze transcript content for the segment."""
        try:
            if not transcript_segment:
                return {"error": "No transcript data available"}
            
            # Extract text content
            text_content = " ".join([
                item.get("content", item.get("text", ""))
                for item in transcript_segment
            ])
            
            # Basic transcript analysis
            word_count = len(text_content.split())
            total_duration = sum([
                item.get("end", 0) - item.get("start", 0)
                for item in transcript_segment
            ])
            
            # Extract key phrases (simple approach)
            words = text_content.lower().split()
            word_freq = {}
            for word in words:
                if len(word) > 3:  # Only consider words longer than 3 characters
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get most frequent words as key phrases
            key_phrases = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return {
                "text_content": text_content,
                "word_count": word_count,
                "duration": total_duration,
                "speech_rate": word_count / total_duration if total_duration > 0 else 0,
                "key_phrases": [phrase[0] for phrase in key_phrases],
                "segments_count": len(transcript_segment),
                "transcript_segments": transcript_segment
            }
            
        except Exception as e:
            logger.error(f"Error analyzing transcript: {e}")
            return {"error": str(e)}
    
    def _analyze_visual_content(
        self, 
        video_path: str, 
        frames_dir: str, 
        start_time: float, 
        end_time: float
    ) -> Dict[str, Any]:
        """Analyze visual content by extracting and analyzing frames."""
        try:
            # Extract frames from the segment
            frame_paths = extract_segment_frames(
                video_path, frames_dir, start_time, end_time, num_frames=3
            )
            
            if not frame_paths:
                return {"error": "No frames could be extracted"}
            
            # Analyze each frame with Gemini
            frame_analyses = []
            for i, frame_path in enumerate(frame_paths):
                try:
                    # Create prompt for technical screen analysis
                    prompt = """Analyze this screenshot from a technical session. Describe:
1. What software/applications are visible
2. What technical activity is being performed
3. Any code, commands, or technical content visible
4. User interface elements and their state
5. Overall technical context and purpose

Provide a concise technical analysis focusing on the educational/instructional content."""
                    
                    # Analyze with Gemini
                    from PIL import Image
                    image = Image.open(frame_path)
                    analysis = analyze_with_gemini(prompt, image)
                    
                    frame_analyses.append({
                        "frame_number": i + 1,
                        "frame_path": frame_path,
                        "timestamp": start_time + (i * (end_time - start_time) / (len(frame_paths) - 1)),
                        "analysis": analysis
                    })
                    
                except Exception as e:
                    logger.error(f"Error analyzing frame {frame_path}: {e}")
                    frame_analyses.append({
                        "frame_number": i + 1,
                        "frame_path": frame_path,
                        "error": str(e)
                    })
            
            # Create visual summary
            visual_summary = self._create_visual_summary(frame_analyses)
            
            return {
                "frames_extracted": len(frame_paths),
                "frame_analyses": frame_analyses,
                "visual_summary": visual_summary,
                "frames_dir": frames_dir
            }
            
        except Exception as e:
            logger.error(f"Error in visual analysis: {e}")
            return {"error": str(e)}
    
    def _analyze_audio_content(self, audio_path: str) -> Dict[str, Any]:
        """Analyze audio content (simplified version)."""
        try:
            if not os.path.exists(audio_path):
                return {"error": "Audio file not found"}
            
            # Get basic audio file information
            file_size = os.path.getsize(audio_path)
            
            # Basic audio analysis (could be extended with librosa or similar)
            return {
                "file_path": audio_path,
                "file_size": file_size,
                "analysis_type": "basic",
                "note": "Advanced audio analysis requires additional dependencies"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing audio: {e}")
            return {"error": str(e)}
    
    def _create_visual_summary(self, frame_analyses: List[Dict[str, Any]]) -> str:
        """Create a summary of visual analysis across frames."""
        if not frame_analyses:
            return "No visual analysis available"
        
        # Extract key information from frame analyses
        software_mentions = set()
        activities = set()
        
        for frame in frame_analyses:
            if "analysis" in frame and not frame.get("error"):
                analysis_text = frame["analysis"].lower()
                
                # Extract software mentions (simple pattern matching)
                common_software = [
                    "vscode", "visual studio", "terminal", "bash", "python", "javascript",
                    "chrome", "firefox", "git", "github", "docker", "kubernetes"
                ]
                
                for software in common_software:
                    if software in analysis_text:
                        software_mentions.add(software)
                
                # Extract activities (simple pattern matching)
                common_activities = [
                    "coding", "debugging", "testing", "configuring", "installing",
                    "running", "executing", "browsing", "editing", "reviewing"
                ]
                
                for activity in common_activities:
                    if activity in analysis_text:
                        activities.add(activity)
        
        # Create summary
        summary_parts = []
        if software_mentions:
            summary_parts.append(f"Software detected: {', '.join(software_mentions)}")
        if activities:
            summary_parts.append(f"Activities observed: {', '.join(activities)}")
        
        return "; ".join(summary_parts) if summary_parts else "General technical content observed"
    
    def _create_multimodal_summary(
        self, 
        transcript_analysis: Dict[str, Any],
        visual_analysis: Dict[str, Any],
        audio_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a comprehensive multimodal summary."""
        try:
            summary = {
                "analysis_timestamp": time.time(),
                "modalities_analyzed": [],
                "key_insights": [],
                "technical_elements": [],
                "confidence_score": 0.0
            }
            
            # Analyze transcript content
            if transcript_analysis and not transcript_analysis.get("error"):
                summary["modalities_analyzed"].append("transcript")
                
                text_content = transcript_analysis.get("text_content", "")
                key_phrases = transcript_analysis.get("key_phrases", [])
                
                if text_content:
                    summary["key_insights"].append(
                        f"Transcript analysis: {len(text_content.split())} words, "
                        f"key topics: {', '.join(key_phrases[:3])}"
                    )
                
                # Extract technical elements from transcript
                technical_terms = [phrase for phrase in key_phrases if any(
                    tech in phrase.lower() for tech in ["code", "command", "install", "run", "error", "config"]
                )]
                summary["technical_elements"].extend(technical_terms)
            
            # Analyze visual content
            if visual_analysis and not visual_analysis.get("error"):
                summary["modalities_analyzed"].append("visual")
                
                visual_summary = visual_analysis.get("visual_summary", "")
                frames_count = visual_analysis.get("frames_extracted", 0)
                
                if visual_summary:
                    summary["key_insights"].append(f"Visual analysis: {visual_summary}")
                    
                    # Extract technical elements from visual analysis
                    if "software detected:" in visual_summary.lower():
                        software_part = visual_summary.lower().split("software detected:")[1].split(";")[0]
                        summary["technical_elements"].extend([
                            s.strip() for s in software_part.split(",") if s.strip()
                        ])
            
            # Analyze audio content
            if audio_analysis and not audio_analysis.get("error"):
                summary["modalities_analyzed"].append("audio")
                summary["key_insights"].append("Audio analysis: Basic file information available")
            
            # Calculate confidence score
            modality_count = len(summary["modalities_analyzed"])
            if modality_count >= 2:
                summary["confidence_score"] = 0.8
            elif modality_count == 1:
                summary["confidence_score"] = 0.6
            else:
                summary["confidence_score"] = 0.3
            
            # Remove duplicate technical elements
            summary["technical_elements"] = list(set(summary["technical_elements"]))
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating multimodal summary: {e}")
            return {"error": str(e)}
    
    def _save_analysis_results(
        self, 
        paths: Dict[str, str], 
        analysis_results: Dict[str, Any], 
        segment_num: int
    ) -> None:
        """Save analysis results to files."""
        try:
            # Save main multimodal analysis
            multimodal_path = paths["multimodal_analysis"]
            with open(multimodal_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_results, f, indent=2, ensure_ascii=False, default=str)
            
            # Save speaker-attributed transcript (simplified version)
            transcript_path = paths["speaker_transcript"]
            transcript_data = analysis_results.get("transcript_analysis", {}).get("transcript_segments", [])
            
            # Add basic speaker information (simplified)
            speaker_attributed = []
            for item in transcript_data:
                attributed_item = item.copy()
                attributed_item["speaker"] = "speaker_1"  # Simplified - would need actual speaker detection
                attributed_item["confidence"] = 0.7
                speaker_attributed.append(attributed_item)
            
            with open(transcript_path, 'w', encoding='utf-8') as f:
                json.dump(speaker_attributed, f, indent=2, ensure_ascii=False, default=str)
            
            # Save segment summary
            summary_path = paths["segment_summary"]
            segment_summary = {
                "segment_number": segment_num,
                "topic": analysis_results["segment_info"]["topic"],
                "duration": analysis_results["segment_info"]["duration"],
                "key_insights": analysis_results["multimodal_summary"].get("key_insights", []),
                "technical_elements": analysis_results["multimodal_summary"].get("technical_elements", []),
                "confidence_score": analysis_results["multimodal_summary"].get("confidence_score", 0.0),
                "modalities_analyzed": analysis_results["multimodal_summary"].get("modalities_analyzed", []),
                "files_generated": {
                    "video_segment": paths["video_file"],
                    "audio_segment": paths["audio_file"],
                    "frames_directory": paths["frames_dir"]
                }
            }
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(segment_summary, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Saved analysis results for segment {segment_num}")
            
        except Exception as e:
            logger.error(f"Error saving analysis results: {e}")


def create_speaker_attributed_transcript(
    transcript_segments: List[Dict[str, Any]],
    audio_analysis: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Create speaker-attributed transcript (simplified version).
    
    Args:
        transcript_segments: Raw transcript segments
        audio_analysis: Optional audio analysis results
        
    Returns:
        Speaker-attributed transcript
    """
    attributed_transcript = []
    
    for i, segment in enumerate(transcript_segments):
        attributed_item = segment.copy()
        
        # Simplified speaker attribution (would need actual speaker detection)
        attributed_item["speaker"] = f"speaker_{(i % 2) + 1}"  # Alternate between speakers
        attributed_item["speaker_confidence"] = 0.7
        attributed_item["is_user"] = i % 3 == 0  # Mark every third segment as user
        attributed_item["user_confidence"] = 0.6 if attributed_item["is_user"] else 0.0
        
        attributed_transcript.append(attributed_item)
    
    return attributed_transcript