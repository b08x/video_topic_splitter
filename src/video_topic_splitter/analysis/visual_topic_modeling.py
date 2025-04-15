# analysis/visual_topic_modeling.py
"""Enhanced topic modeling with visual context integration.

This module extends the base TopicAnalyzer to incorporate visual elements
from video frames, allowing for multimodal analysis that correlates transcript
topics with visual content. It provides methods for analyzing visual scenes,
detecting visual topic shifts, and generating enriched topic segments that
include both textual and visual context.
"""

import logging
import os
import json
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import numpy as np
from tqdm import tqdm

from .topic_modeling import TopicAnalyzer
from ..constants import CHECKPOINTS
from ..project import save_checkpoint
from ..prompt_templates import get_visual_topic_prompt

# Setup logger
logger = logging.getLogger(__name__)

class VisualTopicAnalyzer(TopicAnalyzer):
    """Extends TopicAnalyzer to incorporate visual context from video frames.
    
    This class enhances the text-based topic analysis with visual information,
    allowing for multimodal analysis that correlates transcript topics with
    visual elements in the video. It can detect scene changes, identify visual
    topics, and provide a more comprehensive segmentation of video content.
    
    Attributes:
        All attributes from TopicAnalyzer, plus:
        visual_similarity_threshold (float): Threshold for detecting visual scene changes.
        frame_sample_rate (int): Number of frames to sample per segment for visual analysis.
        visual_context_depth (int): How many previous visual contexts to consider.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: int = 5,
        batch_size: int = 5,
        max_concurrent: int = 3,
        similarity_threshold: float = 0.7,
        register: str = "gen-ai",
        visual_similarity_threshold: float = 0.6,
        frame_sample_rate: int = 5,
        visual_context_depth: int = 2
    ):
        """Initialize the VisualTopicAnalyzer.
        
        Args:
            All arguments from TopicAnalyzer, plus:
            visual_similarity_threshold (float, optional): Threshold for determining
                visual scene changes. Defaults to 0.6.
            frame_sample_rate (int, optional): Number of frames to sample per segment
                for visual analysis. Defaults to 5.
            visual_context_depth (int, optional): How many previous visual contexts
                to consider when analyzing a segment. Defaults to 2.
        """
        super().__init__(
            max_retries=max_retries,
            retry_delay=retry_delay,
            batch_size=batch_size,
            max_concurrent=max_concurrent,
            similarity_threshold=similarity_threshold,
            register=register
        )
        self.visual_similarity_threshold = np.clip(visual_similarity_threshold, 0.0, 1.0)
        self.frame_sample_rate = frame_sample_rate
        self.visual_context_depth = visual_context_depth
        logger.info(
            f"VisualTopicAnalyzer initialized: visual_similarity_threshold={visual_similarity_threshold}, "
            f"frame_sample_rate={frame_sample_rate}, visual_context_depth={visual_context_depth}"
        )
    
    async def analyze_segment_with_visuals_async(
        self, 
        current_segment: Dict, 
        visual_frames: List[Dict],
        previous_segment: Optional[Dict] = None,
        previous_visual_contexts: Optional[List[Dict]] = None
    ) -> Dict:
        """Analyzes a text segment with corresponding visual frames asynchronously.
        
        Extends the base analyze_segment_async method to incorporate visual context
        from video frames that correspond to the segment's time range.
        
        Args:
            current_segment (Dict): Dictionary containing the content of the
                current segment under the key 'content', plus 'start' and 'end' times.
            visual_frames (List[Dict]): List of frame dictionaries with 'timestamp',
                'description', and optionally 'features' keys.
            previous_segment (Optional[Dict], optional): Dictionary containing
                the content and analysis results of the preceding segment.
            previous_visual_contexts (Optional[List[Dict]], optional): List of visual
                context dictionaries from previous segments.
                
        Returns:
            Dict: A dictionary containing the enhanced analysis results, including
                'topic', 'keywords', 'relationship', 'confidence', 'visual_topic',
                'visual_keywords', and 'visual_relationship'.
        """
        current_content = current_segment.get("content", "")
        start_time = current_segment.get("start", 0.0)
        end_time = current_segment.get("end", 0.0)
        
        # Filter frames that fall within this segment's time range
        segment_frames = [
            frame for frame in visual_frames 
            if start_time <= frame.get("timestamp", 0.0) <= end_time
        ]
        
        # Sample frames if there are too many
        if len(segment_frames) > self.frame_sample_rate:
            # Evenly sample frames across the segment
            indices = np.linspace(0, len(segment_frames) - 1, self.frame_sample_rate, dtype=int)
            segment_frames = [segment_frames[i] for i in indices]
        
        # Extract visual descriptions
        visual_descriptions = [frame.get("description", "") for frame in segment_frames]
        
        # Prepare previous context
        prev_content = previous_segment.get("content", "") if previous_segment else None
        prev_topic = previous_segment.get("topic", "Unknown") if previous_segment else "None"
        prev_visual_context = ""
        
        if previous_visual_contexts:
            # Limit to the specified depth
            contexts_to_use = previous_visual_contexts[-self.visual_context_depth:]
            prev_visual_context = "\n".join([
                f"Time {ctx.get('start_time', 0.0)}-{ctx.get('end_time', 0.0)}: "
                f"{ctx.get('visual_description', 'No description')}"
                for ctx in contexts_to_use
            ])
        
        # Create cache keys
        cache_key_current = f"{current_content}|{','.join(visual_descriptions)}"
        cache_key_prev = f"{prev_content}|{prev_visual_context}" if prev_content else None
        
        # Check cache first
        cached_result = self._get_cached_analysis(cache_key_current, cache_key_prev)
        if cached_result:
            logger.debug("Cache hit for visual segment analysis.")
            return cached_result
        
        logger.debug("Cache miss. Analyzing segment with visuals via API.")
        async with self.semaphore:
            # Prepare context with both text and visual information
            context = ""
            if previous_segment:
                max_prev_len = 500
                truncated_prev_content = (
                    (prev_content[:max_prev_len] + "...")
                    if len(prev_content) > max_prev_len
                    else prev_content
                )
                context = (
                    f"Previous segment context:\n"
                    f"Content: {truncated_prev_content}\n"
                    f"Identified Topic: {prev_topic}\n"
                )
                
                if prev_visual_context:
                    context += f"Previous Visual Context:\n{prev_visual_context}\n\n"
                else:
                    context += "\n"
            
            # Add current segment information
            max_curr_len = 1500
            truncated_current_content = (
                (current_content[:max_curr_len] + "...")
                if len(current_content) > max_curr_len
                else current_content
            )
            context += f"Current segment text:\n{truncated_current_content}\n\n"
            
            # Add visual descriptions
            if visual_descriptions:
                context += "Current segment visual descriptions:\n"
                for i, desc in enumerate(visual_descriptions):
                    context += f"Frame {i+1}: {desc}\n"
            else:
                context += "No visual descriptions available for this segment.\n"
            
            # Get the appropriate prompt template for visual analysis
            prompt = get_visual_topic_prompt(self.register, context)
            
            # Similar retry logic as in the parent class
            for attempt in range(self.max_retries):
                try:
                    logger.debug(f"Attempt {attempt + 1}/{self.max_retries} calling LLM for visual analysis...")
                    completion = await self.async_client.chat.completions.create(
                        model="microsoft/phi-4",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=200,  # Increased for visual context
                        response_format={"type": "json_object"},
                    )
                    
                    response_text = completion.choices[0].message.content
                    logger.debug(f"LLM Raw Response for visual analysis: {response_text}")
                    
                    try:
                        result = json.loads(response_text)
                        
                        # Validate expected keys for visual analysis
                        expected_keys = [
                            "topic", "keywords", "relationship", "confidence",
                            "visual_topic", "visual_keywords", "visual_relationship"
                        ]
                        
                        for key in expected_keys:
                            if key not in result:
                                logger.warning(f"LLM response missing expected key '{key}'")
                                if key in ["topic", "visual_topic"]:
                                    result[key] = "Unknown"
                                elif key in ["keywords", "visual_keywords"]:
                                    result[key] = []
                                elif key in ["relationship", "visual_relationship"]:
                                    result[key] = "UNKNOWN"
                                elif key == "confidence":
                                    result[key] = 0
                        
                        # Add visual summary
                        if "visual_summary" not in result:
                            result["visual_summary"] = "No visual summary provided."
                        
                        return result
                        
                    except json.JSONDecodeError as json_e:
                        logger.error(f"Failed to parse JSON response: {json_e}. Response: '{response_text}'")
                        return {
                            "topic": "Parsing Error",
                            "keywords": [],
                            "relationship": "UNKNOWN",
                            "confidence": 0,
                            "visual_topic": "Parsing Error",
                            "visual_keywords": [],
                            "visual_relationship": "UNKNOWN",
                            "visual_summary": "Failed to parse response.",
                            "error": f"JSONDecodeError: {json_e}",
                            "raw_response": response_text
                        }
                    except Exception as parse_e:
                        logger.error(f"Error processing LLM response: {parse_e}. Response: '{response_text}'")
                        return {
                            "topic": "Processing Error",
                            "keywords": [],
                            "relationship": "UNKNOWN",
                            "confidence": 0,
                            "visual_topic": "Processing Error",
                            "visual_keywords": [],
                            "visual_relationship": "UNKNOWN",
                            "visual_summary": "Error processing response.",
                            "error": f"ProcessingError: {parse_e}",
                            "raw_response": response_text
                        }
                        
                except Exception as e:
                    logger.error(f"Unexpected error during API call (Attempt {attempt + 1}): {e}", exc_info=True)
                    if attempt >= self.max_retries - 1:
                        logger.error("Maximum retries reached. Visual analysis failed.")
                        raise
                    await asyncio.sleep(self.retry_delay)
            
            # If all retries fail
            logger.error("Visual segment analysis failed after all retries.")
            return {
                "topic": "Analysis Failed",
                "keywords": [],
                "relationship": "UNKNOWN",
                "confidence": 0,
                "visual_topic": "Analysis Failed",
                "visual_keywords": [],
                "visual_relationship": "UNKNOWN",
                "visual_summary": "Analysis failed after multiple attempts.",
                "error": "Max retries exceeded",
            }
    
    def analyze_segment_with_visuals(
        self, 
        current_segment: Dict, 
        visual_frames: List[Dict],
        previous_segment: Optional[Dict] = None,
        previous_visual_contexts: Optional[List[Dict]] = None
    ) -> Dict:
        """Synchronous wrapper for analyze_segment_with_visuals_async.
        
        Args:
            current_segment (Dict): Dictionary for the current segment.
            visual_frames (List[Dict]): List of frame dictionaries.
            previous_segment (Optional[Dict], optional): Dictionary for the previous segment.
            previous_visual_contexts (Optional[List[Dict]], optional): List of previous visual contexts.
            
        Returns:
            Dict: The analysis result dictionary.
        """
        logger.debug("Running synchronous wrapper for analyze_segment_with_visuals_async.")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return asyncio.run(self.analyze_segment_with_visuals_async(
                    current_segment, visual_frames, previous_segment, previous_visual_contexts
                ))
            else:
                return loop.run_until_complete(
                    self.analyze_segment_with_visuals_async(
                        current_segment, visual_frames, previous_segment, previous_visual_contexts
                    )
                )
        except RuntimeError as e:
            logger.warning(f"RuntimeError during asyncio execution in sync wrapper: {e}. Trying new event loop policy.")
            return asyncio.run(self.analyze_segment_with_visuals_async(
                current_segment, visual_frames, previous_segment, previous_visual_contexts
            ))
    
    def _detect_visual_scene_changes(self, visual_frames: List[Dict]) -> List[int]:
        """Detects significant visual scene changes in a sequence of frames.
        
        Uses visual features or descriptions to identify points where the visual
        content changes significantly, which may indicate topic transitions.
        
        Args:
            visual_frames (List[Dict]): List of frame dictionaries with 'features'
                or 'description' keys.
                
        Returns:
            List[int]: Indices of frames where significant visual changes occur.
        """
        if not visual_frames or len(visual_frames) < 2:
            return []
        
        scene_change_indices = []
        
        # Check if frames have feature vectors
        if "features" in visual_frames[0]:
            # Use feature vectors for similarity calculation
            for i in range(1, len(visual_frames)):
                prev_features = visual_frames[i-1].get("features")
                curr_features = visual_frames[i].get("features")
                
                if prev_features is not None and curr_features is not None:
                    # Convert to numpy arrays if they aren't already
                    if not isinstance(prev_features, np.ndarray):
                        prev_features = np.array(prev_features)
                    if not isinstance(curr_features, np.ndarray):
                        curr_features = np.array(curr_features)
                    
                    # Calculate cosine similarity
                    similarity = np.dot(prev_features, curr_features) / (
                        np.linalg.norm(prev_features) * np.linalg.norm(curr_features)
                    )
                    
                    if similarity < self.visual_similarity_threshold:
                        scene_change_indices.append(i)
        else:
            # Use text descriptions for similarity
            for i in range(1, len(visual_frames)):
                prev_desc = visual_frames[i-1].get("description", "")
                curr_desc = visual_frames[i].get("description", "")
                
                similarity = self._calculate_similarity(prev_desc, curr_desc)
                
                if similarity < self.visual_similarity_threshold:
                    scene_change_indices.append(i)
        
        return scene_change_indices
    
    async def _analyze_visual_batches(
        self, 
        batches: List[List[Dict]], 
        visual_frames: List[Dict]
    ) -> List[Dict]:
        """Analyzes batches with visual context.
        
        Similar to _analyze_batches but incorporates visual information.
        
        Args:
            batches (List[List[Dict]]): List of sentence batches.
            visual_frames (List[Dict]): List of visual frame dictionaries.
            
        Returns:
            List[Dict]: List of analysis results with visual context.
        """
        analyses = [None] * len(batches)
        previous_analysis_result = None
        previous_visual_contexts = []
        
        with tqdm(total=len(batches), desc="Analyzing Batches with Visuals", unit="batch") as pbar:
            for i, batch in enumerate(batches):
                combined_segment = self._combine_batch(batch)
                if not combined_segment:
                    logger.warning(f"Skipping empty batch at index {i}")
                    analyses[i] = None
                    pbar.update(1)
                    continue
                
                # Find frames that correspond to this segment's time range
                start_time = combined_segment.get("start", 0.0)
                end_time = combined_segment.get("end", 0.0)
                
                # Analyze with visual context
                analysis_result = await self.analyze_segment_with_visuals_async(
                    combined_segment, 
                    visual_frames,
                    previous_analysis_result, 
                    previous_visual_contexts
                )
                analyses[i] = analysis_result
                
                # Update previous context for next iteration
                previous_analysis_result = {
                    "content": combined_segment.get("content", ""),
                    "topic": analysis_result.get("topic", "Unknown"),
                }
                
                # Add to visual contexts
                visual_context = {
                    "start_time": start_time,
                    "end_time": end_time,
                    "visual_topic": analysis_result.get("visual_topic", "Unknown"),
                    "visual_description": analysis_result.get("visual_summary", "")
                }
                previous_visual_contexts.append(visual_context)
                
                pbar.update(1)
        
        return analyses
    
    def identify_segments_with_visuals(
        self, 
        transcript_sentences: List[Dict], 
        visual_frames: List[Dict]
    ) -> List[Dict]:
        """Identifies topic-based segments incorporating visual information.
        
        Extends identify_segments to consider both transcript content and
        visual elements when determining segment boundaries.
        
        Args:
            transcript_sentences (List[Dict]): List of sentence dictionaries.
            visual_frames (List[Dict]): List of visual frame dictionaries.
            
        Returns:
            List[Dict]: List of identified segments with visual context.
        """
        if not transcript_sentences:
            logger.warning("Transcript sentences list is empty. Cannot identify segments.")
            return []
        
        logger.info("Identifying segments based on combined text and visual analysis...")
        
        # 1. Create batches
        logger.info("Creating analysis batches...")
        batches = self._create_batches(transcript_sentences)
        if not batches:
            logger.warning("No batches were created from the transcript sentences.")
            return []
        logger.info(f"Created {len(batches)} batches.")
        
        # 2. Analyze batches with visual context
        logger.info("Analyzing batches using LLM with visual context...")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                analyses = asyncio.run(self._analyze_visual_batches(batches, visual_frames))
            else:
                analyses = loop.run_until_complete(self._analyze_visual_batches(batches, visual_frames))
        except RuntimeError:
            logger.warning("RuntimeError during asyncio execution. Trying asyncio.run().")
            analyses = asyncio.run(self._analyze_visual_batches(batches, visual_frames))
        
        if len(analyses) != len(batches):
            logger.error(f"Mismatch between number of batches ({len(batches)}) and analyses ({len(analyses)}).")
            return []
        
        # 3. Process analyses to form segments
        logger.info("Merging batch analyses into final segments with visual context...")
        segments = []
        current_segment_batches = []
        current_segment_analysis = None
        
        for i, (batch, analysis) in enumerate(zip(batches, analyses)):
            if analysis is None:
                logger.warning(f"Skipping batch {i+1} due to missing analysis.")
                continue
            
            if not current_segment_batches:
                # Start the first segment
                current_segment_batches.extend(batch)
                current_segment_analysis = analysis
            else:
                # Consider both text and visual relationships
                text_relationship = analysis.get("relationship", "UNKNOWN").upper()
                visual_relationship = analysis.get("visual_relationship", "UNKNOWN").upper()
                confidence = analysis.get("confidence", 0)
                
                # Define thresholds
                new_segment_threshold = 70
                shift_segment_threshold = 85
                
                # Determine if we should split based on text or visual cues
                should_split = False
                split_reason = ""
                
                # Check text relationship
                if text_relationship == "NEW" and confidence > new_segment_threshold:
                    should_split = True
                    split_reason = f"NEW text relationship, confidence {confidence}"
                elif text_relationship == "SHIFT" and confidence > shift_segment_threshold:
                    should_split = True
                    split_reason = f"SHIFT text relationship, confidence {confidence}"
                
                # Check visual relationship if we haven't decided to split yet
                if not should_split:
                    if visual_relationship == "NEW" and confidence > new_segment_threshold:
                        should_split = True
                        split_reason = f"NEW visual relationship, confidence {confidence}"
                    elif visual_relationship == "SHIFT" and confidence > shift_segment_threshold:
                        should_split = True
                        split_reason = f"SHIFT visual relationship, confidence {confidence}"
                
                if should_split:
                    # Finalize the current segment
                    finalized_segment = self._combine_batch(current_segment_batches)
                    # Add topic information
                    finalized_segment["topic"] = current_segment_analysis.get("topic", "Unknown")
                    finalized_segment["keywords"] = current_segment_analysis.get("keywords", [])
                    # Add visual information
                    finalized_segment["visual_topic"] = current_segment_analysis.get("visual_topic", "Unknown")
                    finalized_segment["visual_keywords"] = current_segment_analysis.get("visual_keywords", [])
                    finalized_segment["visual_summary"] = current_segment_analysis.get("visual_summary", "")
                    
                    segments.append(finalized_segment)
                    logger.debug(f"Finalized segment {len(segments)} ending at batch {i}, split reason: {split_reason}")
                    
                    # Start a new segment
                    current_segment_batches = list(batch)
                    current_segment_analysis = analysis
                else:
                    # Continue the current segment
                    current_segment_batches.extend(batch)
        
        # Handle the last segment
        if current_segment_batches:
            finalized_segment = self._combine_batch(current_segment_batches)
            if current_segment_analysis:
                finalized_segment["topic"] = current_segment_analysis.get("topic", "Unknown")
                finalized_segment["keywords"] = current_segment_analysis.get("keywords", [])
                finalized_segment["visual_topic"] = current_segment_analysis.get("visual_topic", "Unknown")
                finalized_segment["visual_keywords"] = current_segment_analysis.get("visual_keywords", [])
                finalized_segment["visual_summary"] = current_segment_analysis.get("visual_summary", "")
            else:
                finalized_segment["topic"] = "Unknown"
                finalized_segment["keywords"] = []
                finalized_segment["visual_topic"] = "Unknown"
                finalized_segment["visual_keywords"] = []
                finalized_segment["visual_summary"] = ""
            
            segments.append(finalized_segment)
            logger.debug(f"Finalized last segment {len(segments)}")
        
        logger.info(f"Identified {len(segments)} topic segments with visual context.")
        return segments


def process_transcript_with_visuals(
    transcript_sentences: List[Dict],
    visual_frames: List[Dict],
    project_path: str,
    register: str = "gen-ai",
) -> Dict:
    """Processes a transcript with visual frames to perform multimodal topic analysis.
    
    Uses the VisualTopicAnalyzer to identify segments based on both transcript
    content and visual elements. Formats results into a dictionary with metadata
    about identified topics, segments, and visual context.
    
    Args:
        transcript_sentences (List[Dict]): List of sentence dictionaries.
        visual_frames (List[Dict]): List of visual frame dictionaries.
        project_path (str): Path to the project directory.
        register (str, optional): Analysis register/domain. Defaults to "gen-ai".
        
    Returns:
        Dict: Dictionary containing analysis results with visual context.
    """
    logger.info(f"Starting transcript processing with visual context for project: {project_path}")
    results_path = os.path.join(project_path, "visual_topic_analysis_results.json")
    
    # Check if results already exist
    if os.path.exists(results_path):
        logger.info(f"Loading existing visual topic analysis results from: {results_path}")
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            return results
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load existing results file ({e}). Proceeding with analysis.")
    
    # Initialize the analyzer
    try:
        analyzer = VisualTopicAnalyzer(register=register)
    except ValueError as ve:
        logger.error(f"Failed to initialize VisualTopicAnalyzer: {ve}")
        raise
    
    # Identify segments using the analyzer
    logger.info("Identifying topic segments with visual context...")
    try:
        segments = analyzer.identify_segments_with_visuals(transcript_sentences, visual_frames)
    except Exception as e:
        logger.error(f"Error during segment identification with visuals: {e}", exc_info=True)
        raise
    
    if not segments:
        logger.warning("No segments were identified. Returning empty results.")
        return {"topics": [], "visual_topics": [], "segments": [], "register": register}
    
    # Generate metadata and structure results
    logger.info("Formatting analysis results with visual context...")
    segment_metadata = []
    topic_summary = {}  # Text topics
    visual_topic_summary = {}  # Visual topics
    
    for i, segment in enumerate(tqdm(segments, desc="Generating Segment Metadata")):
        segment_id = i + 1
        start_time = segment.get("start", 0.0)
        end_time = segment.get("end", 0.0)
        topic = segment.get("topic", "Unknown")
        keywords = segment.get("keywords", [])
        visual_topic = segment.get("visual_topic", "Unknown")
        visual_keywords = segment.get("visual_keywords", [])
        visual_summary = segment.get("visual_summary", "")
        content = segment.get("content", "")
        
        segment_meta = {
            "segment_id": segment_id,
            "start_time": start_time,
            "end_time": end_time,
            "duration": max(0.0, end_time - start_time),
            "dominant_topic": topic,
            "top_keywords": keywords,
            "visual_topic": visual_topic,
            "visual_keywords": visual_keywords,
            "visual_summary": visual_summary,
            "transcript": content,
        }
        segment_metadata.append(segment_meta)
        
        # Update topic summaries
        if topic != "Unknown":
            if topic not in topic_summary:
                topic_summary[topic] = set()
            topic_summary[topic].update(keywords)
        
        if visual_topic != "Unknown":
            if visual_topic not in visual_topic_summary:
                visual_topic_summary[visual_topic] = set()
            visual_topic_summary[visual_topic].update(visual_keywords)
    
    # Create lists of unique topics
    topics_list = [
        {"topic_id": i + 1, "topic": topic_name, "words": sorted(list(kw_set))}
        for i, (topic_name, kw_set) in enumerate(topic_summary.items())
    ]
    
    visual_topics_list = [
        {"topic_id": i + 1, "topic": topic_name, "words": sorted(list(kw_set))}
        for i, (topic_name, kw_set) in enumerate(visual_topic_summary.items())
    ]
    
    # Create final results structure
    results = {
        "topics": topics_list,
        "visual_topics": visual_topics_list,
        "segments": segment_metadata,
        "register": register,
        "analysis_info": {
            "total_segments": len(segments),
            "analyzer_config": {
                "register": analyzer.register,
                "batch_size": analyzer.batch_size,
                "similarity_threshold": analyzer.similarity_threshold,
                "visual_similarity_threshold": analyzer.visual_similarity_threshold,
                "frame_sample_rate": analyzer.frame_sample_rate,
            }
        }
    }
    
    # Save results to JSON file
    logger.info(f"Saving visual topic analysis results to: {results_path}")
    try:
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    except IOError as e:
        logger.error(f"Failed to save results to {results_path}: {e}")
    except TypeError as e:
        logger.error(f"Failed to serialize results to JSON: {e}")
    
    # Save checkpoint
    logger.info("Saving visual topic modeling checkpoint...")
    save_checkpoint(
        project_path,
        CHECKPOINTS["VISUAL_TOPIC_MODELING_COMPLETE"],
        {"results_path": results_path, "num_segments": len(segments)}
    )
    
    logger.info("Transcript processing with visual context complete.")
    return results