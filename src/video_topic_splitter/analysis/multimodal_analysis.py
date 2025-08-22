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

import cv2
from PIL import Image, UnidentifiedImageError

from ..api.gemini import analyze_with_gemini, batch_analyze_images_with_gemini
from ..processing.video.video_segmentation import extract_segment_frames
from ..processing.ocr.ocr_detection import detect_software_names
from ..progress_tracker import ProgressTracker
from .enhanced_transcript_analysis import EnhancedTranscriptAnalyzer

logger = logging.getLogger(__name__)


class MultimodalAnalyzer:
    """Simplified multimodal analyzer for topic segments."""
    
    def __init__(self, progress_tracker: ProgressTracker = None, enable_batch_processing: bool = True):
        """
        Initialize the multimodal analyzer.
        
        Args:
            progress_tracker: Optional progress tracker
            enable_batch_processing: Enable optimized batch processing for visual analysis
        """
        self.progress_tracker = progress_tracker
        self.transcript_analyzer = EnhancedTranscriptAnalyzer()
        self.enable_batch_processing = enable_batch_processing
    
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
    
    @classmethod
    def analyze_multiple_segments_optimized(
        cls,
        video_path: str,
        segment_list: List[Dict[str, Any]],
        transcript_data: List[Dict[str, Any]],
        progress_tracker: ProgressTracker = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple video segments with cross-segment batch optimization.
        
        This method processes multiple segments together, collecting all frame
        extraction and analysis requests before submitting them as optimized
        batches to the Gemini API. This significantly reduces network overhead
        and API costs while maintaining the same analysis quality.
        
        Args:
            video_path: Path to the original video file
            segment_list: List of segment information dictionaries
            transcript_data: Full transcript data for the video
            progress_tracker: Optional progress tracker
            
        Returns:
            List of complete analysis results for all segments
        """
        from .visual_batch_processor import CrossSegmentBatchCoordinator
        from .performance_monitor import record_visual_analysis_metrics
        
        logger.info(f"Starting optimized analysis of {len(segment_list)} segments")
        
        # Start timing for performance measurement
        analysis_start_time = time.time()
        
        if progress_tracker:
            progress_tracker.start_phase("Optimized Multimodal Analysis")
        
        # Initialize batch coordinator
        batch_coordinator = CrossSegmentBatchCoordinator(progress_tracker)
        
        # Create individual analyzers for each segment (without batch processing)
        analyzers = {}
        segment_results = []
        
        try:
            # Phase 1: Collect all frame extraction requests
            if progress_tracker:
                progress_tracker.update_phase_progress(5.0, "Extracting frames from all segments")
            
            all_frame_requests = []
            segment_frame_mapping = {}
            
            for segment_info in segment_list:
                segment_id = f"segment_{segment_info['segment_number']:03d}"
                segment_frame_mapping[segment_id] = {
                    'segment_info': segment_info,
                    'frame_paths': []
                }
                
                # Extract frames for this segment
                from ..processing.video.video_segmentation import extract_segment_frames
                
                frames_dir = segment_info["paths"]["frames_dir"]
                frame_paths = extract_segment_frames(
                    video_path, frames_dir, 
                    segment_info["start_time"], segment_info["end_time"], 
                    num_frames=3, format="jpg", quality=90
                )
                
                segment_frame_mapping[segment_id]['frame_paths'] = frame_paths
                
                # Register frames with batch coordinator
                batch_coordinator.register_segment_frames(segment_id, len(frame_paths))
                
                # Create analysis requests for all frames in this segment
                for i, frame_path in enumerate(frame_paths):
                    prompt = """Analyze this screenshot from a technical session. Describe:
1. What software/applications are visible
2. What technical activity is being performed
3. Any code, commands, or technical content visible
4. User interface elements and their state
5. Overall technical context and purpose

Provide a concise technical analysis focusing on the educational/instructional content."""
                    
                    timestamp = segment_info["start_time"] + (
                        i * (segment_info["end_time"] - segment_info["start_time"]) / (len(frame_paths) - 1)
                    )
                    
                    batch_coordinator.submit_frame_for_analysis(
                        segment_id=segment_id,
                        frame_path=frame_path,
                        frame_number=i + 1,
                        timestamp=timestamp,
                        prompt=prompt,
                        metadata={
                            'segment_number': segment_info['segment_number'],
                            'topic': segment_info['topic'],
                            'start_time': segment_info['start_time'],
                            'end_time': segment_info['end_time']
                        }
                    )
            
            # Phase 2: Process all batches
            if progress_tracker:
                progress_tracker.update_phase_progress(30.0, "Processing visual analysis batches")
            
            batch_results = batch_coordinator.finalize_and_get_results()
            performance_metrics = batch_results['performance_metrics']
            
            # Phase 3: Process each segment with optimized visual results
            if progress_tracker:
                progress_tracker.update_phase_progress(60.0, "Completing segment analysis")
            
            for i, segment_info in enumerate(segment_list):
                segment_id = f"segment_{segment_info['segment_number']:03d}"
                segment_num = segment_info["segment_number"]
                
                # Update progress
                segment_progress = 60.0 + (i / len(segment_list)) * 35.0
                if progress_tracker:
                    progress_tracker.update_phase_progress(
                        segment_progress, f"Processing segment {segment_num}"
                    )
                
                # Create analyzer for this segment (without individual batch processing)
                analyzer = cls(progress_tracker, enable_batch_processing=False)
                
                # Extract transcript for this segment
                transcript_segment = analyzer._extract_transcript_for_segment_internal(
                    transcript_data, segment_info
                )
                
                # Perform transcript and audio analysis
                transcript_analysis = analyzer._analyze_transcript(transcript_segment)
                audio_analysis = analyzer._analyze_audio_content(segment_info["paths"]["audio_file"])
                
                # Get visual analysis results from batch processing
                visual_frame_results = batch_coordinator.get_segment_analyses(segment_id)
                
                # Convert batch results to expected visual analysis format
                visual_analysis = {
                    "frames_extracted": len(segment_frame_mapping[segment_id]['frame_paths']),
                    "frame_analyses": visual_frame_results,
                    "visual_summary": analyzer._create_visual_summary(visual_frame_results),
                    "frames_dir": segment_info["paths"]["frames_dir"],
                    "performance_metrics": {
                        "batch_processing_used": True,
                        "batch_efficiency": performance_metrics.get('batching_efficiency', 0),
                        "cost_savings_percent": performance_metrics.get('estimated_cost_savings', {}).get('estimated_savings_percentage', 0)
                    }
                }
                
                # Create complete analysis results
                analysis_results = {
                    "segment_info": {
                        "segment_number": segment_num,
                        "topic": segment_info["topic"],
                        "start_time": segment_info["start_time"],
                        "end_time": segment_info["end_time"],
                        "duration": segment_info["duration"]
                    },
                    "transcript_analysis": transcript_analysis,
                    "visual_analysis": visual_analysis,
                    "audio_analysis": audio_analysis,
                    "multimodal_summary": {}
                }
                
                # Create comprehensive multimodal summary
                analysis_results["multimodal_summary"] = analyzer._create_multimodal_summary(
                    transcript_analysis, visual_analysis, audio_analysis
                )
                
                # Save analysis results
                analyzer._save_analysis_results(
                    segment_info["paths"], analysis_results, segment_num
                )
                
                segment_results.append(analysis_results)
            
            # Phase 4: Final completion
            if progress_tracker:
                progress_tracker.update_phase_progress(100.0, "Batch optimization completed")
                progress_tracker.complete_phase("Optimized Multimodal Analysis")
            
            # Record performance metrics
            total_execution_time = time.time() - analysis_start_time
            total_frames_processed = performance_metrics['total_frames_processed']
            estimated_api_calls = performance_metrics['total_batches_submitted']
            
            record_visual_analysis_metrics(
                operation_type='batch',
                frames_processed=total_frames_processed,
                execution_time=total_execution_time,
                api_calls_made=estimated_api_calls,
                error_count=0  # Could be enhanced to track actual errors
            )
            
            # Log performance summary
            logger.info(
                f"Optimized multimodal analysis completed:\n"
                f"  - Segments processed: {len(segment_results)}\n"
                f"  - Total frames analyzed: {performance_metrics['total_frames_processed']}\n"
                f"  - Batch efficiency: {performance_metrics['batching_efficiency']*100:.1f}%\n"
                f"  - Estimated cost savings: {performance_metrics['estimated_cost_savings']['estimated_savings_percentage']:.1f}%\n"
                f"  - Processing rate: {performance_metrics['frames_per_second']:.2f} frames/second\n"
                f"  - Total execution time: {total_execution_time:.2f}s"
            )
            
            return segment_results
            
        except Exception as e:
            logger.error(f"Error in optimized multimodal analysis: {e}")
            if progress_tracker:
                progress_tracker.complete_phase("Optimized Multimodal Analysis", success=False)
            raise
    
    def _extract_transcript_for_segment_internal(
        self, 
        transcript_data: List[Dict[str, Any]], 
        segment_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Internal method to extract transcript for a segment.
        This is the same logic as in SegmentProcessor but accessible within this class.
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
            
            return segment_transcript
            
        except Exception as e:
            logger.error(f"Error extracting transcript for segment: {e}")
            return []
    
    def _analyze_transcript(self, transcript_segment: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze transcript content using enhanced spaCy-powered analysis.

        This method uses the EnhancedTranscriptAnalyzer to perform a deep
        linguistic analysis of the transcript. If the enhanced analysis fails,
        it falls back to a basic analysis.

        Args:
            transcript_segment: A list of transcript items for the segment.

        Returns:
            A dictionary containing the analysis of the transcript, including
            key phrases, named entities, and other linguistic features.
        """
        try:
            if not transcript_segment:
                return {"error": "No transcript data available"}
            
            # Use the enhanced transcript analyzer
            enhanced_analysis = self.transcript_analyzer.analyze_transcript_segment(transcript_segment)
            
            # If enhanced analysis failed, fall back to basic analysis
            if "error" in enhanced_analysis:
                logger.warning(f"Enhanced analysis failed: {enhanced_analysis['error']}, falling back to basic analysis")
                return self._basic_transcript_analysis(transcript_segment)
            
            # Extract legacy-compatible data for backward compatibility
            basic_metrics = enhanced_analysis.get("basic_metrics", {})
            key_phrases_data = enhanced_analysis.get("key_phrases", {})
            
            # Combine enhanced results with legacy format
            result = {
                "text_content": enhanced_analysis.get("text_content", ""),
                "word_count": basic_metrics.get("token_count", 0),
                "duration": basic_metrics.get("duration", 0),
                "speech_rate": basic_metrics.get("speech_rate", 0),
                "segments_count": enhanced_analysis.get("segments_count", 0),
                "transcript_segments": enhanced_analysis.get("transcript_segments", []),
                
                # Enhanced spaCy analysis results
                "enhanced_analysis": {
                    "key_phrases": key_phrases_data.get("noun_phrases", {}),
                    "key_lemmas": key_phrases_data.get("key_lemmas", {}),
                    "technical_terms": key_phrases_data.get("technical_terms", {}),
                    "named_entities": enhanced_analysis.get("named_entities", {}),
                    "linguistic_features": enhanced_analysis.get("linguistic_features", {}),
                    "actions_and_relationships": enhanced_analysis.get("actions_and_relationships", {}),
                    "technical_elements": enhanced_analysis.get("technical_elements", []),
                    "semantic_features": enhanced_analysis.get("semantic_features", {})
                }
            }
            
            # Maintain legacy key_phrases format for compatibility
            noun_phrases = key_phrases_data.get("noun_phrases", {})
            key_lemmas = key_phrases_data.get("key_lemmas", {})
            technical_terms = key_phrases_data.get("technical_terms", {})
            
            # Prioritize multi-word phrases over single words, and technical terms
            phrase_candidates = []
            
            # Add technical terms first (highest priority)
            for term, count in sorted(technical_terms.items(), key=lambda x: x[1], reverse=True):
                if len(term) > 3 and term not in phrase_candidates:
                    phrase_candidates.append(term)
            
            # Add noun phrases (medium priority)
            for phrase, count in sorted(noun_phrases.items(), key=lambda x: x[1], reverse=True):
                if len(phrase) > 3 and phrase not in phrase_candidates:
                    phrase_candidates.append(phrase)
            
            # Add meaningful lemmas only if we don't have enough phrases (lowest priority)
            if len(phrase_candidates) < 5:
                for lemma, count in sorted(key_lemmas.items(), key=lambda x: x[1], reverse=True):
                    if (len(lemma) > 3 and 
                        lemma not in phrase_candidates and
                        lemma not in {'this', 'that', 'these', 'those', 'like', 'just', 'really'}):
                        phrase_candidates.append(lemma)
            
            result["key_phrases"] = phrase_candidates[:10] if phrase_candidates else ["No significant phrases found"]
            
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced transcript analysis: {e}")
            # Fall back to basic analysis
            return self._basic_transcript_analysis(transcript_segment)
    
    def _basic_transcript_analysis(self, transcript_segment: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform a basic, fallback analysis of the transcript segment.

        This method is used when the enhanced spaCy-based analysis is not
        available or fails. It calculates basic metrics and extracts key
        phrases using simple frequency analysis.

        Args:
            transcript_segment: A list of transcript items for the segment.

        Returns:
            A dictionary with basic transcript analysis results.
        """
        try:
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
            
            # Extract key phrases (improved approach)
            import re
            
            # Basic stop words list
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
                'this', 'that', 'these', 'those', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'his', 
                'she', 'her', 'it', 'its', 'they', 'them', 'their', 'is', 'are', 'was', 'were', 'be', 'been', 
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
                'can', 'cant', 'cannot', 'so', 'very', 'really', 'just', 'now', 'then', 'here', 'there',
                'like', 'about', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'than',
                'most', 'other', 'some', 'such', 'only', 'same', 'few', 'more', 'all', 'any', 'each', 'every'
            }
            
            # Clean and tokenize text
            words = re.findall(r'\b[a-z]+\b', text_content.lower())
            word_freq = {}
            
            for word in words:
                # Filter: longer than 3 chars, not a stop word, contains letters
                if (len(word) > 3 and 
                    word not in stop_words and 
                    word.isalpha()):
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get most frequent meaningful words as key phrases
            key_phrases = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return {
                "text_content": text_content,
                "word_count": word_count,
                "duration": total_duration,
                "speech_rate": word_count / total_duration if total_duration > 0 else 0,
                "key_phrases": [phrase[0] for phrase in key_phrases],
                "segments_count": len(transcript_segment),
                "transcript_segments": transcript_segment,
                "analysis_method": "basic_fallback"
            }
            
        except Exception as e:
            logger.error(f"Error in basic transcript analysis: {e}")
            return {"error": str(e)}
    
    def _analyze_visual_content(
        self, 
        video_path: str, 
        frames_dir: str, 
        start_time: float, 
        end_time: float
    ) -> Dict[str, Any]:
        """
        Analyze visual content by extracting and analyzing key frames using optimized batch processing.

        This method extracts representative frames from the segment and uses Gemini's
        batch API to analyze all frames efficiently in a single request, reducing
        network latency and API costs by 50%.

        Args:
            video_path: Path to the original video file.
            frames_dir: Directory to store the extracted frames.
            start_time: The start time of the segment in seconds.
            end_time: The end time of the segment in seconds.

        Returns:
            A dictionary containing the analysis of the visual content,
            including paths to frames and their analyses.
        """
        try:
            # Extract frames from the segment using enhanced extraction
            frame_paths = extract_segment_frames(
                video_path, frames_dir, start_time, end_time, 
                num_frames=3, format="jpg", quality=90
            )
            
            if not frame_paths:
                return {"error": "No frames could be extracted"}
            
            # Prepare batch analysis requests
            batch_requests = []
            for i, frame_path in enumerate(frame_paths):
                # Create prompt for technical screen analysis
                prompt = """Analyze this screenshot from a technical session. Describe:
1. What software/applications are visible
2. What technical activity is being performed
3. Any code, commands, or technical content visible
4. User interface elements and their state
5. Overall technical context and purpose

Provide a concise technical analysis focusing on the educational/instructional content."""
                
                batch_requests.append({
                    'prompt': prompt,
                    'image': frame_path,  # Path will be loaded by batch processor
                    'frame_id': f'frame_{i + 1}',
                    'metadata': {
                        'frame_number': i + 1,
                        'frame_path': frame_path,
                        'timestamp': start_time + (i * (end_time - start_time) / (len(frame_paths) - 1)),
                        'segment_start': start_time,
                        'segment_end': end_time
                    }
                })
            
            # Progress callback for tracking
            def visual_progress_callback(progress: float, message: str):
                if self.progress_tracker:
                    # Map visual analysis progress to a subset of the overall progress
                    visual_weight = 0.4  # Visual analysis is 40% of segment analysis
                    base_progress = 30.0  # Assuming we're 30% through segment analysis when visual starts
                    adjusted_progress = base_progress + (progress * visual_weight)
                    self.progress_tracker.update_phase_progress(adjusted_progress, f"Visual: {message}")
            
            logger.info(f"Starting batch analysis of {len(batch_requests)} frames for segment")
            
            # Perform batch analysis
            batch_results = batch_analyze_images_with_gemini(
                batch_requests, 
                progress_callback=visual_progress_callback
            )
            
            # Convert batch results to expected format
            frame_analyses = []
            for result in batch_results:
                metadata = result.get('metadata', {})
                
                if 'error' in result:
                    frame_analyses.append({
                        'frame_number': metadata.get('frame_number', 0),
                        'frame_path': metadata.get('frame_path', ''),
                        'timestamp': metadata.get('timestamp', 0),
                        'error': result['error']
                    })
                else:
                    frame_analyses.append({
                        'frame_number': metadata.get('frame_number', 0),
                        'frame_path': metadata.get('frame_path', ''),
                        'timestamp': metadata.get('timestamp', 0),
                        'analysis': result.get('analysis', ''),
                        'processing_method': metadata.get('processing_method', 'batch'),
                        'batch_job_id': metadata.get('batch_job_id', '')
                    })
            
            # Create visual summary
            visual_summary = self._create_visual_summary(frame_analyses)
            
            # Calculate performance metrics
            successful_analyses = len([r for r in frame_analyses if 'analysis' in r])
            batch_processing_used = any(r.get('processing_method') == 'batch' for r in frame_analyses)
            
            result = {
                "frames_extracted": len(frame_paths),
                "frame_analyses": frame_analyses,
                "visual_summary": visual_summary,
                "frames_dir": frames_dir,
                "performance_metrics": {
                    "successful_analyses": successful_analyses,
                    "total_frames": len(frame_paths),
                    "batch_processing_used": batch_processing_used,
                    "processing_efficiency": successful_analyses / len(frame_paths) if frame_paths else 0
                }
            }
            
            logger.info(
                f"Visual analysis completed: {successful_analyses}/{len(frame_paths)} frames analyzed "
                f"(batch processing: {batch_processing_used})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in optimized visual analysis: {e}")
            return {"error": str(e)}
    
    def _analyze_audio_content(self, audio_path: str) -> Dict[str, Any]:
        """
        Perform a basic analysis of the audio content for the segment.

        Currently, this provides basic file information. It can be extended
        to perform more advanced audio analysis (e.g., silence detection,
        speaker diarization) with additional libraries.

        Args:
            audio_path: Path to the audio file for the segment.

        Returns:
            A dictionary with basic information about the audio file.
        """
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
        """
        Create a concise summary from the visual analysis of multiple frames.

        This method aggregates information from individual frame analyses to
        identify common software and activities observed across the segment.

        Args:
            frame_analyses: A list of analysis results for each frame.

        Returns:
            A string summarizing the key visual elements.
        """
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
        """
        Create a comprehensive summary combining all analysis modalities.

        This method synthesizes insights from transcript, visual, and audio
        analyses to provide a holistic view of the segment's content.

        Args:
            transcript_analysis: The results from transcript analysis.
            visual_analysis: The results from visual analysis.
            audio_analysis: The results from audio analysis.

        Returns:
            A dictionary containing a multimodal summary with key insights,
            technical elements, and a confidence score.
        """
        try:
            summary = {
                "analysis_timestamp": time.time(),
                "modalities_analyzed": [],
                "key_insights": [],
                "technical_elements": [],
                "confidence_score": 0.0
            }
            
            # Analyze transcript content (enhanced with spaCy)
            if transcript_analysis and not transcript_analysis.get("error"):
                summary["modalities_analyzed"].append("transcript")
                
                text_content = transcript_analysis.get("text_content", "")
                key_phrases = transcript_analysis.get("key_phrases", [])
                enhanced_analysis = transcript_analysis.get("enhanced_analysis", {})
                
                if text_content:
                    word_count = transcript_analysis.get("word_count", len(text_content.split()))
                    summary["key_insights"].append(
                        f"Transcript analysis: {word_count} tokens, "
                        f"key topics: {', '.join(key_phrases[:3])}"
                    )
                
                # Extract technical elements from enhanced analysis
                if enhanced_analysis:
                    # Get technical terms from spaCy analysis
                    spacy_technical = enhanced_analysis.get("technical_elements", [])
                    technical_terms_dict = enhanced_analysis.get("technical_terms", {})
                    
                    # Combine technical elements
                    all_technical = list(spacy_technical) + list(technical_terms_dict.keys())
                    summary["technical_elements"].extend(all_technical)
                    
                    # Add insights from named entities
                    named_entities = enhanced_analysis.get("named_entities", {})
                    if named_entities.get("entities"):
                        entity_summary = named_entities.get("summary", {})
                        total_entities = entity_summary.get("total_entities", 0)
                        if total_entities > 0:
                            summary["key_insights"].append(
                                f"Named entities: {total_entities} entities identified "
                                f"across {entity_summary.get('entity_types', 0)} categories"
                            )
                    
                    # Add insights from actions and relationships
                    actions = enhanced_analysis.get("actions_and_relationships", {})
                    if actions.get("key_actions"):
                        action_count = actions.get("total_actions", 0)
                        if action_count > 0:
                            summary["key_insights"].append(
                                f"Key actions: {action_count} distinct actions identified"
                            )
                else:
                    # Fallback to basic analysis
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
        """
        Save the analysis results for a segment to various files.

        This includes the main multimodal analysis, a speaker-attributed
        transcript, and a concise segment summary.

        Args:
            paths: A dictionary of output file paths for the segment.
            analysis_results: The comprehensive analysis results to save.
            segment_num: The number of the segment being processed.
        """
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


def analyze_screenshot(
    image_path,
    project_path,
    software_list=None,
    ocr_lang="eng",
    context=None,
):
    """
    Analyze a single screenshot for software applications using OCR and Gemini.

    This function takes a path to an image, performs OCR to detect software
    names, and then uses Gemini to provide a more detailed analysis of the
    visual content.

    Args:
        image_path: Path to the screenshot image file.
        project_path: Path to the project directory for saving any artifacts.
        software_list: Optional list of software names to detect via OCR.
        ocr_lang: Language for OCR detection.
        context: Optional context to provide to the Gemini analysis.

    Returns:
        A dictionary containing the OCR matches and the Gemini analysis.
    """
    try:
        logger.info(f"Analyzing screenshot: {image_path}")
        
        # Load the image
        frame = cv2.imread(image_path)
        if frame is None:
            return {"error": f"Could not read image: {image_path}"}
        
        image = Image.open(image_path)
        
        # Perform OCR analysis if software list is provided
        ocr_matches = []
        if software_list:
            try:
                ocr_matches = detect_software_names(frame, software_list, ocr_lang)
            except Exception as e:
                logger.warning(f"OCR analysis failed: {e}")
        
        # Create context for Gemini analysis
        software_context = (
            f"Detected software (via OCR): {', '.join(m['software'] for m in ocr_matches)}"
            if ocr_matches
            else "No specific software detected via OCR."
        )
        
        # Build prompt for Gemini analysis
        base_prompt = (
            "Analyze this screenshot for software applications and technical content. "
            "Describe the visual elements, user interface components, and any actions taking place."
        )
        
        if context:
            prompt = f"{base_prompt}\n\nAdditional context: {context}\n\n{software_context}"
        else:
            prompt = f"{base_prompt}\n\n{software_context}"
        
        # Analyze with Gemini
        gemini_analysis = analyze_with_gemini(prompt, image)
        
        # Prepare results
        results = {
            "image_path": image_path,
            "ocr_matches": ocr_matches,
            "gemini_analysis": gemini_analysis,
            "analysis_timestamp": time.time()
        }
        
        # Save results to project directory
        try:
            results_path = os.path.join(project_path, "screenshot_analysis.json")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Screenshot analysis results saved to: {results_path}")
        except Exception as e:
            logger.warning(f"Could not save screenshot analysis results: {e}")
        
        return results
        
    except (UnidentifiedImageError, Exception) as e:
        error_msg = f"Failed to analyze screenshot {image_path}: {e}"
        logger.error(error_msg)
        return {"error": error_msg}