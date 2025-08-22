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
from ..utils.prompts import load_and_process_template, get_default_template_path, load_prompt_template

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
            "transcript_analysis": self._analyze_transcript(transcript_segment, segment_info),
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
                transcript_analysis = analyzer._analyze_transcript(transcript_segment, segment_info)
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
    
    def _summarize_transcript_segment(self, transcript_segment: List[Dict[str, Any]], context: str) -> str:
        """
        Generate concise, context-aware summaries of transcript segments using the SFL framework.
        
        This method loads the SFL subtitle summary prompt template, substitutes the transcript
        text and context placeholders, and calls the Gemini LLM service to generate a summary.
        
        Args:
            transcript_segment: List of transcript items for the segment
            context: Contextual information to inform the summary generation
            
        Returns:
            A concise summary of the transcript segment, or an error message if generation fails
        """
        try:
            if not transcript_segment:
                return "No transcript content available for summary generation"
            
            # Extract the transcript text content
            transcript_text = " ".join([
                item.get("content", item.get("text", ""))
                for item in transcript_segment
            ])
            
            if not transcript_text.strip():
                return "Empty transcript content - no summary generated"
            
            # Load and process the SFL prompt template
            try:
                template_path = get_default_template_path("sfl_subtitle_summary_prompt")
                
                # Define template variables for substitution
                template_variables = {
                    "SUBTITLE_TEXT": transcript_text,
                    "CONTEXT": context
                }
                
                # Load and process the template with variable substitution
                prompt = load_and_process_template(template_path, template_variables)
                logger.debug("Successfully loaded and processed SFL subtitle summary prompt")
                
            except FileNotFoundError:
                logger.error("SFL subtitle summary prompt template not found")
                return "Error: SFL prompt template not found"
            except Exception as e:
                logger.error(f"Error loading SFL prompt template: {e}")
                return f"Error loading prompt template: {str(e)}"
            
            # Call Gemini LLM service to generate the summary
            try:
                logger.info("Generating transcript summary using Gemini LLM")
                summary = analyze_with_gemini(prompt, image=None)  # Text-only analysis
                
                if not summary or summary.strip() == "":
                    return "LLM generated empty summary"
                
                logger.debug(f"Successfully generated transcript summary of {len(summary)} characters")
                return summary.strip()
                
            except Exception as e:
                logger.error(f"Error calling Gemini for transcript summary: {e}")
                return f"Error generating summary: {str(e)}"
                
        except Exception as e:
            logger.error(f"Unexpected error in _summarize_transcript_segment: {e}")
            return f"Unexpected error: {str(e)}"

    def _analyze_transcript(self, transcript_segment: List[Dict[str, Any]], segment_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze transcript content using enhanced spaCy-powered analysis.

        This method uses the EnhancedTranscriptAnalyzer to perform a deep
        linguistic analysis of the transcript. If the enhanced analysis fails,
        it falls back to a basic analysis. Additionally generates an SFL summary.

        Args:
            transcript_segment: A list of transcript items for the segment.
            segment_info: Segment metadata including topic, timing, and other context.

        Returns:
            A dictionary containing the analysis of the transcript, including
            key phrases, named entities, other linguistic features, and SFL summary.
        """
        try:
            if not transcript_segment:
                return {"error": "No transcript data available"}
            
            # Use the enhanced transcript analyzer
            enhanced_analysis = self.transcript_analyzer.analyze_transcript_segment(transcript_segment)
            
            # If enhanced analysis failed, fall back to basic analysis
            if "error" in enhanced_analysis:
                logger.warning(f"Enhanced analysis failed: {enhanced_analysis['error']}, falling back to basic analysis")
                return self._basic_transcript_analysis(transcript_segment, segment_info)
            
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
            
            # Generate SFL-based summary for the transcript segment
            try:
                # Build context information for the SFL summary
                context_parts = [
                    f"Video Topic: {segment_info.get('topic', 'Unknown')}",
                    f"Segment: {segment_info.get('segment_number', 'N/A')} of total video",
                    f"Duration: {segment_info.get('duration', 0):.1f} seconds",
                    f"Timeframe: {segment_info.get('start_time', 0):.1f}s - {segment_info.get('end_time', 0):.1f}s"
                ]
                
                # Add key phrases for additional context
                if phrase_candidates:
                    context_parts.append(f"Key Topics: {', '.join(phrase_candidates[:5])}")
                
                context_info = "; ".join(context_parts)
                
                # Generate the SFL summary
                sfl_summary = self._summarize_transcript_segment(transcript_segment, context_info)
                result["sfl_summary"] = sfl_summary
                
                logger.debug(f"Successfully generated SFL summary for segment {segment_info.get('segment_number', 'N/A')}")
                
            except Exception as e:
                logger.warning(f"Failed to generate SFL summary: {e}")
                result["sfl_summary"] = f"Summary generation failed: {str(e)}"
            
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced transcript analysis: {e}")
            # Fall back to basic analysis
            return self._basic_transcript_analysis(transcript_segment, segment_info)
    
    def _basic_transcript_analysis(self, transcript_segment: List[Dict[str, Any]], segment_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a basic, fallback analysis of the transcript segment.

        This method is used when the enhanced spaCy-based analysis is not
        available or fails. It calculates basic metrics and extracts key
        phrases using simple frequency analysis. Also includes SFL summary.

        Args:
            transcript_segment: A list of transcript items for the segment.
            segment_info: Segment metadata including topic, timing, and other context.

        Returns:
            A dictionary with basic transcript analysis results and SFL summary.
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
            phrase_list = [phrase[0] for phrase in key_phrases]
            
            # Generate SFL-based summary for the transcript segment (basic context)
            try:
                # Build basic context information for the SFL summary
                context_parts = [
                    f"Video Topic: {segment_info.get('topic', 'Unknown')}",
                    f"Segment: {segment_info.get('segment_number', 'N/A')} of total video",
                    f"Duration: {segment_info.get('duration', 0):.1f} seconds",
                    f"Timeframe: {segment_info.get('start_time', 0):.1f}s - {segment_info.get('end_time', 0):.1f}s"
                ]
                
                # Add key phrases for additional context
                if phrase_list:
                    context_parts.append(f"Key Topics: {', '.join(phrase_list[:5])}")
                
                context_info = "; ".join(context_parts)
                
                # Generate the SFL summary
                sfl_summary = self._summarize_transcript_segment(transcript_segment, context_info)
                
                logger.debug(f"Successfully generated SFL summary for segment {segment_info.get('segment_number', 'N/A')} (basic analysis)")
                
            except Exception as e:
                logger.warning(f"Failed to generate SFL summary in basic analysis: {e}")
                sfl_summary = f"Summary generation failed: {str(e)}"
            
            return {
                "text_content": text_content,
                "word_count": word_count,
                "duration": total_duration,
                "speech_rate": word_count / total_duration if total_duration > 0 else 0,
                "key_phrases": phrase_list,
                "segments_count": len(transcript_segment),
                "transcript_segments": transcript_segment,
                "sfl_summary": sfl_summary,
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
        Create a comprehensive summary using the SFL multimodal technical analysis framework.

        This method replaces the heuristic-based approach with a sophisticated LLM-driven
        SFL framework that synthesizes insights from transcript, visual, and audio analyses
        through structured prompting and Gemini API integration.

        Args:
            transcript_analysis: The results from transcript analysis.
            visual_analysis: The results from visual analysis.
            audio_analysis: The results from audio analysis.

        Returns:
            A dictionary containing a multimodal summary generated by the SFL framework
            with structured insights and comprehensive technical analysis.
        """
        try:
            # Gather visual summary and transcript summary for SFL prompt
            visual_summary = self._extract_visual_analysis_summary(visual_analysis)
            transcript_summary = self._extract_transcript_analysis_summary(transcript_analysis)
            
            # Load the SFL multimodal technical analysis prompt
            try:
                template_path = get_default_template_path("sfl_multimodal_technical_analysis_prompt")
                
                # Load the SFL framework prompt template
                sfl_framework_prompt = load_prompt_template(template_path)
                
                # Append the actual analysis data to the SFL framework prompt
                full_prompt = f"""{sfl_framework_prompt}

---

## CURRENT SESSION DATA FOR ANALYSIS

### Visual Analysis Summary
{visual_summary}

### Transcript Analysis Summary
{transcript_summary}

### Analysis Request
Please analyze the above visual and transcript data using the SFL multimodal technical analysis framework. Provide comprehensive insights that synthesize both modalities according to the framework specifications above.

Focus on:
1. **Multimodal Technical State**: What comprehensive picture emerges from both visual interface evidence and spoken technical content?
2. **Cross-Modal Correlation**: How do the visual actions align with or contradict the spoken content?
3. **Enhanced Issue Detection**: What problems can be identified through the combination of both evidence sources?
4. **Comprehensive Workflow Analysis**: What does the synthesis reveal about technical progress and competence?
5. **Multimodal Recommendations**: What actions should be taken based on the comprehensive evidence?

Generate your analysis following the structured output format specified in the framework above."""
                
                logger.debug("Successfully loaded and constructed SFL multimodal analysis prompt")
                sfl_prompt = full_prompt
                
            except FileNotFoundError:
                logger.error("SFL multimodal technical analysis prompt template not found")
                return self._fallback_multimodal_summary(transcript_analysis, visual_analysis, audio_analysis)
            except Exception as e:
                logger.error(f"Error loading SFL multimodal prompt template: {e}")
                return self._fallback_multimodal_summary(transcript_analysis, visual_analysis, audio_analysis)
            
            # Get a representative keyframe image for multimodal analysis
            keyframe_image = self._get_representative_keyframe(visual_analysis)
            
            # Call Gemini API with the SFL prompt and keyframe image
            try:
                logger.info("Generating multimodal summary using SFL framework and Gemini API")
                sfl_analysis_result = analyze_with_gemini(sfl_prompt, keyframe_image)
                
                if not sfl_analysis_result or sfl_analysis_result.strip() == "":
                    logger.warning("SFL analysis returned empty result, falling back to heuristic approach")
                    return self._fallback_multimodal_summary(transcript_analysis, visual_analysis, audio_analysis)
                
                # Parse and structure the SFL analysis result
                structured_summary = self._parse_sfl_analysis_output(
                    sfl_analysis_result, transcript_analysis, visual_analysis, audio_analysis
                )
                
                logger.info("Successfully generated SFL-based multimodal summary")
                return structured_summary
                
            except Exception as e:
                logger.error(f"Error calling Gemini API for SFL multimodal analysis: {e}")
                return self._fallback_multimodal_summary(transcript_analysis, visual_analysis, audio_analysis)
                
        except Exception as e:
            logger.error(f"Unexpected error in SFL multimodal summary generation: {e}")
            return self._fallback_multimodal_summary(transcript_analysis, visual_analysis, audio_analysis)
    
    def _extract_visual_analysis_summary(self, visual_analysis: Dict[str, Any]) -> str:
        """
        Extract and format visual analysis summary for SFL prompt injection.
        
        Args:
            visual_analysis: Visual analysis results dictionary
            
        Returns:
            Formatted visual analysis summary for SFL template
        """
        if not visual_analysis or visual_analysis.get("error"):
            return "No visual analysis available - unable to extract frames or perform visual content analysis."
        
        visual_summary_parts = []
        
        # Add frames information
        frames_count = visual_analysis.get("frames_extracted", 0)
        if frames_count > 0:
            visual_summary_parts.append(f"Extracted and analyzed {frames_count} representative frames from the segment.")
        
        # Add visual summary
        visual_summary = visual_analysis.get("visual_summary", "")
        if visual_summary:
            visual_summary_parts.append(f"Visual Content Summary: {visual_summary}")
        
        # Add performance metrics if available
        performance_metrics = visual_analysis.get("performance_metrics", {})
        if performance_metrics:
            successful_analyses = performance_metrics.get("successful_analyses", 0)
            total_frames = performance_metrics.get("total_frames", 0)
            if total_frames > 0:
                success_rate = (successful_analyses / total_frames) * 100
                visual_summary_parts.append(f"Analysis Success Rate: {success_rate:.1f}% ({successful_analyses}/{total_frames} frames)")
        
        # Add individual frame analyses if available
        frame_analyses = visual_analysis.get("frame_analyses", [])
        if frame_analyses:
            technical_content_found = []
            for frame in frame_analyses:
                if "analysis" in frame and not frame.get("error"):
                    analysis_text = frame["analysis"]
                    # Extract key technical points (first 150 chars as preview)
                    preview = analysis_text[:150] + "..." if len(analysis_text) > 150 else analysis_text
                    timestamp = frame.get("timestamp", "unknown")
                    technical_content_found.append(f"Frame at {timestamp:.1f}s: {preview}")
            
            if technical_content_found:
                visual_summary_parts.append("Technical Content Observed:")
                visual_summary_parts.extend([f"- {content}" for content in technical_content_found])
        
        return "\n".join(visual_summary_parts) if visual_summary_parts else "No visual analysis data available."
    
    def _extract_transcript_analysis_summary(self, transcript_analysis: Dict[str, Any]) -> str:
        """
        Extract and format transcript analysis summary for SFL prompt injection.
        
        Args:
            transcript_analysis: Transcript analysis results dictionary
            
        Returns:
            Formatted transcript analysis summary for SFL template
        """
        if not transcript_analysis or transcript_analysis.get("error"):
            return "No transcript analysis available - unable to process audio content."
        
        transcript_summary_parts = []
        
        # Add basic metrics
        word_count = transcript_analysis.get("word_count", 0)
        duration = transcript_analysis.get("duration", 0)
        speech_rate = transcript_analysis.get("speech_rate", 0)
        
        if word_count > 0 and duration > 0:
            transcript_summary_parts.append(
                f"Transcript Analysis: {word_count} words spoken over {duration:.1f} seconds "
                f"(rate: {speech_rate:.1f} words/second)"
            )
        
        # Add key phrases
        key_phrases = transcript_analysis.get("key_phrases", [])
        if key_phrases:
            phrases_preview = ", ".join(key_phrases[:8])  # Show top 8 key phrases
            transcript_summary_parts.append(f"Key Topics Discussed: {phrases_preview}")
        
        # Add SFL summary if available
        sfl_summary = transcript_analysis.get("sfl_summary", "")
        if sfl_summary and "Error" not in sfl_summary and "Failed" not in sfl_summary:
            transcript_summary_parts.append(f"Content Summary: {sfl_summary}")
        
        # Add enhanced analysis insights if available
        enhanced_analysis = transcript_analysis.get("enhanced_analysis", {})
        if enhanced_analysis:
            # Add technical elements
            technical_elements = enhanced_analysis.get("technical_elements", [])
            if technical_elements:
                tech_preview = ", ".join(technical_elements[:6])  # Show top 6 technical elements
                transcript_summary_parts.append(f"Technical Elements: {tech_preview}")
            
            # Add named entities summary
            named_entities = enhanced_analysis.get("named_entities", {})
            entity_summary = named_entities.get("summary", {})
            if entity_summary.get("total_entities", 0) > 0:
                total_entities = entity_summary.get("total_entities")
                entity_types = entity_summary.get("entity_types", 0)
                transcript_summary_parts.append(f"Named Entities: {total_entities} entities across {entity_types} categories")
            
            # Add actions summary
            actions = enhanced_analysis.get("actions_and_relationships", {})
            if actions.get("total_actions", 0) > 0:
                action_count = actions.get("total_actions")
                transcript_summary_parts.append(f"Technical Actions: {action_count} distinct actions identified")
        
        # Add raw text content (truncated)
        text_content = transcript_analysis.get("text_content", "")
        if text_content:
            content_preview = text_content[:200] + "..." if len(text_content) > 200 else text_content
            transcript_summary_parts.append(f"Spoken Content Preview: '{content_preview}'")
        
        return "\n".join(transcript_summary_parts) if transcript_summary_parts else "No transcript analysis data available."
    
    def _get_representative_keyframe(self, visual_analysis: Dict[str, Any]) -> Optional[Image.Image]:
        """
        Get a representative keyframe image for multimodal analysis.
        
        Args:
            visual_analysis: Visual analysis results containing frame information
            
        Returns:
            PIL Image object of the representative keyframe, or None if not available
        """
        try:
            frame_analyses = visual_analysis.get("frame_analyses", [])
            if not frame_analyses:
                logger.warning("No frame analyses available for keyframe selection")
                return None
            
            # Try to find the middle frame as most representative
            middle_index = len(frame_analyses) // 2
            representative_frame = frame_analyses[middle_index]
            
            frame_path = representative_frame.get("frame_path", "")
            if not frame_path or not os.path.exists(frame_path):
                # Fallback to first available frame
                for frame in frame_analyses:
                    frame_path = frame.get("frame_path", "")
                    if frame_path and os.path.exists(frame_path):
                        break
                else:
                    logger.warning("No valid frame paths found in visual analysis")
                    return None
            
            # Load and return the image
            image = Image.open(frame_path)
            logger.debug(f"Successfully loaded keyframe image from {frame_path}")
            return image
            
        except Exception as e:
            logger.error(f"Error loading representative keyframe: {e}")
            return None
    
    def _parse_sfl_analysis_output(
        self,
        sfl_analysis_result: str,
        transcript_analysis: Dict[str, Any],
        visual_analysis: Dict[str, Any],
        audio_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Parse and structure the SFL analysis output to match the expected multimodal_summary format.
        
        Args:
            sfl_analysis_result: Raw text output from the SFL-based Gemini analysis
            transcript_analysis: Original transcript analysis for fallback data
            visual_analysis: Original visual analysis for fallback data
            audio_analysis: Original audio analysis for fallback data
            
        Returns:
            Structured multimodal summary dictionary compatible with downstream processing
        """
        try:
            # Base structure for multimodal summary
            structured_summary = {
                "analysis_timestamp": time.time(),
                "analysis_method": "sfl_framework",
                "modalities_analyzed": [],
                "key_insights": [],
                "technical_elements": [],
                "confidence_score": 0.0,
                "sfl_analysis": sfl_analysis_result,
                "structured_analysis": {}
            }
            
            # Determine which modalities were successfully analyzed
            if transcript_analysis and not transcript_analysis.get("error"):
                structured_summary["modalities_analyzed"].append("transcript")
            if visual_analysis and not visual_analysis.get("error"):
                structured_summary["modalities_analyzed"].append("visual")
            if audio_analysis and not audio_analysis.get("error"):
                structured_summary["modalities_analyzed"].append("audio")
            
            # Parse the SFL analysis for structured insights
            sfl_insights = self._extract_insights_from_sfl_output(sfl_analysis_result)
            structured_summary["key_insights"] = sfl_insights.get("insights", [])
            structured_summary["technical_elements"] = sfl_insights.get("technical_elements", [])
            structured_summary["structured_analysis"] = sfl_insights.get("structured_sections", {})
            
            # Calculate confidence score based on modalities and SFL output quality
            modality_count = len(structured_summary["modalities_analyzed"])
            sfl_quality_score = self._assess_sfl_output_quality(sfl_analysis_result)
            
            # Confidence based on modalities (0.0-0.7) + SFL quality (0.0-0.3)
            base_confidence = min(0.7, modality_count * 0.35)  # Max 0.7 for 2+ modalities
            quality_bonus = sfl_quality_score * 0.3  # Max 0.3 for high quality
            structured_summary["confidence_score"] = min(1.0, base_confidence + quality_bonus)
            
            # Add fallback data if SFL parsing was limited
            if not structured_summary["key_insights"]:
                structured_summary["key_insights"] = self._extract_fallback_insights(
                    transcript_analysis, visual_analysis, audio_analysis
                )
            
            if not structured_summary["technical_elements"]:
                structured_summary["technical_elements"] = self._extract_fallback_technical_elements(
                    transcript_analysis, visual_analysis, audio_analysis
                )
            
            logger.info(f"Successfully parsed SFL analysis with {len(structured_summary['key_insights'])} insights")
            return structured_summary
            
        except Exception as e:
            logger.error(f"Error parsing SFL analysis output: {e}")
            # Fallback to basic structure with SFL content
            return {
                "analysis_timestamp": time.time(),
                "analysis_method": "sfl_framework_with_errors",
                "modalities_analyzed": ["transcript", "visual"] if visual_analysis else ["transcript"],
                "key_insights": [f"SFL Analysis Generated (parsing errors encountered): {sfl_analysis_result[:200]}..."],
                "technical_elements": [],
                "confidence_score": 0.4,
                "sfl_analysis": sfl_analysis_result,
                "parsing_error": str(e)
            }
    
    def _extract_insights_from_sfl_output(self, sfl_output: str) -> Dict[str, Any]:
        """
        Extract structured insights from SFL analysis output using pattern matching.
        
        Args:
            sfl_output: Raw SFL analysis text
            
        Returns:
            Dictionary containing extracted insights and technical elements
        """
        insights = []
        technical_elements = []
        structured_sections = {}
        
        # Split into sections based on common SFL output patterns
        sections = self._split_sfl_sections(sfl_output)
        structured_sections = sections
        
        # Extract insights from different sections
        for section_name, section_content in sections.items():
            if section_name.lower() in ["technical state", "multimodal technical state", "visual evidence"]:
                # Extract technical elements and state information
                tech_items = self._extract_technical_items(section_content)
                technical_elements.extend(tech_items)
                if section_content.strip():
                    insights.append(f"Technical State: {section_content.strip()[:150]}...")
            
            elif section_name.lower() in ["issue detection", "enhanced issue detection", "issues"]:
                # Extract issue-related insights
                if section_content.strip():
                    insights.append(f"Issues Identified: {section_content.strip()[:150]}...")
            
            elif section_name.lower() in ["workflow analysis", "comprehensive workflow analysis"]:
                # Extract workflow insights
                if section_content.strip():
                    insights.append(f"Workflow Analysis: {section_content.strip()[:150]}...")
            
            elif section_name.lower() in ["recommendations", "multimodal recommendations"]:
                # Extract recommendations
                if section_content.strip():
                    insights.append(f"Recommendations: {section_content.strip()[:150]}...")
        
        # If no structured sections found, extract general insights
        if not insights:
            # Look for bullet points, numbered lists, or paragraph structure
            general_insights = self._extract_general_insights(sfl_output)
            insights.extend(general_insights)
        
        # Extract technical elements from the full text if none found in sections
        if not technical_elements:
            technical_elements = self._extract_technical_items(sfl_output)
        
        return {
            "insights": insights[:10],  # Limit to top 10 insights
            "technical_elements": list(set(technical_elements))[:15],  # Unique, limit to 15
            "structured_sections": structured_sections
        }
    
    def _split_sfl_sections(self, text: str) -> Dict[str, str]:
        """Split SFL output into sections based on headers and structure."""
        sections = {}
        
        # Look for markdown-style headers
        lines = text.split('\n')
        current_section = "general"
        current_content = []
        
        for line in lines:
            # Check for section headers (## or **bold** patterns)
            if line.strip().startswith('##') or (line.strip().startswith('**') and line.strip().endswith('**')):
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = line.strip().replace('##', '').replace('**', '').strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Save final section
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def _extract_technical_items(self, text: str) -> List[str]:
        """Extract technical items from text using pattern matching."""
        import re
        
        technical_elements = []
        
        # Common technical patterns
        patterns = [
            r'\b(?:VS Code|Visual Studio|PyCharm|IntelliJ|Eclipse|Atom|Sublime)\b',  # IDEs
            r'\b(?:Python|JavaScript|Java|C\+\+|Ruby|PHP|Go|Rust|TypeScript)\b',   # Languages
            r'\b(?:Git|Docker|Kubernetes|Jenkins|Travis|CircleCI)\b',             # DevOps tools
            r'\b(?:React|Vue|Angular|Django|Flask|Spring|Express)\b',             # Frameworks
            r'\b(?:MySQL|PostgreSQL|MongoDB|Redis|SQLite)\b',                    # Databases
            r'\b(?:AWS|Azure|GCP|Heroku|Netlify|Vercel)\b',                     # Cloud platforms
            r'\b(?:terminal|command line|CLI|shell|bash|zsh)\b',                 # Terminal references
            r'\b(?:API|REST|GraphQL|JSON|XML|YAML)\b',                          # API/Data formats
            r'\b(?:error|bug|debug|test|build|deploy|install|configure)\b',     # Actions
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            technical_elements.extend([match.lower() for match in matches])
        
        return technical_elements
    
    def _extract_general_insights(self, text: str) -> List[str]:
        """Extract general insights from unstructured SFL output."""
        import re
        
        insights = []
        
        # Split by sentences and look for meaningful statements
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            # Look for insights (sentences with technical content and reasonable length)
            if (len(sentence) > 30 and len(sentence) < 200 and 
                any(keyword in sentence.lower() for keyword in 
                    ['technical', 'analysis', 'observed', 'visible', 'detected', 'identified', 
                     'shows', 'indicates', 'suggests', 'workflow', 'process', 'system'])):
                insights.append(sentence)
        
        return insights[:8]  # Limit to 8 general insights
    
    def _assess_sfl_output_quality(self, sfl_output: str) -> float:
        """
        Assess the quality of SFL output for confidence scoring.
        
        Returns a score between 0.0 and 1.0 based on output characteristics.
        """
        if not sfl_output or len(sfl_output.strip()) < 50:
            return 0.0
        
        quality_score = 0.0
        
        # Length-based quality (0.0-0.3)
        if len(sfl_output) > 500:
            quality_score += 0.3
        elif len(sfl_output) > 200:
            quality_score += 0.2
        else:
            quality_score += 0.1
        
        # Structure-based quality (0.0-0.4)
        if '##' in sfl_output or '**' in sfl_output:  # Has headers
            quality_score += 0.2
        if any(marker in sfl_output for marker in ['•', '-', '1.', '2.', '3.']):  # Has lists
            quality_score += 0.2
        
        # Content-based quality (0.0-0.3)
        technical_keywords = ['software', 'interface', 'technical', 'analysis', 'workflow', 'system']
        keyword_count = sum(1 for keyword in technical_keywords if keyword.lower() in sfl_output.lower())
        quality_score += min(0.3, keyword_count * 0.05)
        
        return min(1.0, quality_score)
    
    def _extract_fallback_insights(
        self, 
        transcript_analysis: Dict[str, Any], 
        visual_analysis: Dict[str, Any], 
        audio_analysis: Dict[str, Any]
    ) -> List[str]:
        """Extract basic insights as fallback when SFL parsing fails."""
        insights = []
        
        # From transcript
        if transcript_analysis and not transcript_analysis.get("error"):
            word_count = transcript_analysis.get("word_count", 0)
            key_phrases = transcript_analysis.get("key_phrases", [])
            if word_count > 0:
                insights.append(f"Transcript: {word_count} words, key topics: {', '.join(key_phrases[:3])}")
        
        # From visual
        if visual_analysis and not visual_analysis.get("error"):
            visual_summary = visual_analysis.get("visual_summary", "")
            if visual_summary:
                insights.append(f"Visual: {visual_summary}")
        
        return insights
    
    def _extract_fallback_technical_elements(
        self, 
        transcript_analysis: Dict[str, Any], 
        visual_analysis: Dict[str, Any], 
        audio_analysis: Dict[str, Any]
    ) -> List[str]:
        """Extract basic technical elements as fallback when SFL parsing fails."""
        technical_elements = []
        
        # From transcript
        if transcript_analysis and not transcript_analysis.get("error"):
            key_phrases = transcript_analysis.get("key_phrases", [])
            enhanced_analysis = transcript_analysis.get("enhanced_analysis", {})
            
            if enhanced_analysis:
                spacy_technical = enhanced_analysis.get("technical_elements", [])
                technical_elements.extend(spacy_technical)
            else:
                # Basic extraction
                tech_phrases = [phrase for phrase in key_phrases if any(
                    tech in phrase.lower() for tech in ["code", "command", "install", "run", "error", "config"]
                )]
                technical_elements.extend(tech_phrases)
        
        return technical_elements
    
    def _fallback_multimodal_summary(
        self, 
        transcript_analysis: Dict[str, Any],
        visual_analysis: Dict[str, Any],
        audio_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fallback to heuristic-based multimodal summary when SFL approach fails.
        
        This preserves the original heuristic logic as a backup when SFL processing
        encounters errors or the prompt template is unavailable.
        """
        logger.info("Using fallback heuristic-based multimodal summary")
        
        try:
            summary = {
                "analysis_timestamp": time.time(),
                "analysis_method": "heuristic_fallback",
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
            logger.error(f"Error in fallback multimodal summary: {e}")
            return {"error": str(e), "analysis_method": "fallback_failed"}
    
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


def analyze_screenshot_sfl(
    image_path: str,
    project_path: str,
    context: Optional[str] = None,
    software_list: Optional[List[str]] = None,
    ocr_lang: str = "eng"
) -> Dict[str, Any]:
    """
    Analyze a single screenshot using the SFL (Systemic Functional Linguistics) framework.
    
    This function provides enhanced technical screenshot analysis using a structured
    prompt template that follows SFL principles for comprehensive technical session
    analysis, including interface identification, state assessment, workflow analysis,
    and actionable recommendations.
    
    Args:
        image_path: Path to the screenshot image file.
        project_path: Path to the project directory for saving any artifacts.
        context: Optional context to provide additional information about the session.
        software_list: Optional list of software names to detect via OCR.
        ocr_lang: Language for OCR detection (default: "eng").
        
    Returns:
        A dictionary containing the comprehensive SFL-based analysis results.
    """
    try:
        logger.info(f"Starting SFL analysis of screenshot: {image_path}")
        
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
                logger.debug(f"OCR detected {len(ocr_matches)} software matches")
            except Exception as e:
                logger.warning(f"OCR analysis failed: {e}")
        
        # Prepare context for SFL prompt
        sfl_context_parts = []
        
        # Add provided context
        if context:
            sfl_context_parts.append(f"Session Context: {context}")
        
        # Add OCR findings
        if ocr_matches:
            software_names = [match['software'] for match in ocr_matches]
            sfl_context_parts.append(f"Detected Software (via OCR): {', '.join(software_names)}")
        else:
            sfl_context_parts.append("No specific software detected via OCR analysis.")
        
        # Add image metadata
        image_info = {
            "file_path": image_path,
            "file_size": os.path.getsize(image_path),
            "image_dimensions": f"{image.width}x{image.height}",
            "image_mode": image.mode
        }
        sfl_context_parts.append(f"Image Information: {image_info['image_dimensions']} {image_info['image_mode']} image")
        
        # Combine all context
        combined_context = "\n".join(sfl_context_parts)
        
        # Load and process the SFL prompt template
        try:
            template_path = get_default_template_path("sfl_technical_screenshot_analysis_prompt")
            
            # Define template variables
            template_variables = {
                "CONTEXT": combined_context,
                "IMAGE_PATH": image_path,
                "PROJECT_PATH": project_path,
                "TIMESTAMP": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            
            # Load and process the template
            sfl_prompt = load_and_process_template(template_path, template_variables)
            logger.debug("Successfully loaded and processed SFL prompt template")
            
        except FileNotFoundError:
            logger.error("SFL prompt template not found, falling back to built-in prompt")
            # Fallback prompt if template file is missing
            sfl_prompt = f"""
# SFL Technical Screenshot Analysis

Analyze this screenshot from a technical session following the SFL framework principles.

## Context Information:
{combined_context}

## Analysis Requirements:

### Interface Identification (Field)
- Identify all visible technical tools, applications, and interfaces
- Assess current states of technical processes and systems
- Evaluate apparent workflow progress and development phase

### Technical Assessment (Tenor)  
- **Interface Interpreter**: Precise identification of visible tools and their states
- **Workflow Analyst**: Strategic analysis of development progress and session flow
- **Troubleshooting Guide**: Identify issues and recommend resolution approaches
- **Session Facilitator**: Provide guidance for maintaining productive workflow

### Structured Analysis (Mode)
Provide analysis in the following structure:

1. **Technical State Summary**:
   - Visible Tools: List applications and interfaces with current status
   - Active Processes: Observable technical processes with progress indicators
   - System Health: Assessment of visible performance and resource usage

2. **Issue Assessment**:
   - Critical Issues: Immediate blockers requiring attention
   - Warnings: Potential problems that may impact progress
   - Optimization Opportunities: Observed inefficiencies or improvements

3. **Workflow Analysis**:
   - Current Phase: Apparent development stage (coding, testing, debugging, deployment)
   - Progress Indicators: Evidence of forward movement or completion status
   - Next Logical Steps: Recommended actions based on observed state

4. **Contextual Recommendations**:
   - Immediate Actions: Specific steps to address visible issues
   - Tool Suggestions: Recommended tools or interface adjustments
   - Workflow Optimization: Suggestions for improving session efficiency

Focus on evidence-based analysis using visible interface elements and provide actionable insights for technical session support.
            """
        
        # Analyze with Gemini using the SFL-enhanced prompt
        logger.info("Submitting image for SFL-based Gemini analysis")
        gemini_analysis = analyze_with_gemini(sfl_prompt, image)
        
        # Prepare comprehensive results
        results = {
            "analysis_type": "sfl_framework",
            "image_path": image_path,
            "image_info": image_info,
            "context_provided": context,
            "ocr_matches": ocr_matches,
            "sfl_analysis": gemini_analysis,
            "analysis_timestamp": time.time(),
            "analysis_datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
            "template_used": "sfl_technical_screenshot_analysis_prompt",
            "processing_details": {
                "ocr_enabled": bool(software_list),
                "ocr_language": ocr_lang,
                "software_detected_count": len(ocr_matches),
                "context_sections": len(sfl_context_parts)
            }
        }
        
        # Save results to project directory with SFL-specific naming
        try:
            results_filename = "sfl_screenshot_analysis.json"
            results_path = os.path.join(project_path, results_filename)
            
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"SFL screenshot analysis results saved to: {results_path}")
            results["results_file"] = results_path
            
        except Exception as e:
            logger.warning(f"Could not save SFL screenshot analysis results: {e}")
        
        logger.info("SFL screenshot analysis completed successfully")
        return results
        
    except (UnidentifiedImageError, Exception) as e:
        error_msg = f"Failed to perform SFL analysis of screenshot {image_path}: {e}"
        logger.error(error_msg)
        return {
            "analysis_type": "sfl_framework", 
            "error": error_msg,
            "image_path": image_path,
            "analysis_timestamp": time.time()
        }