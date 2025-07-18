#!/usr/bin/env python3
"""Topic modeling and transcript segmentation functionality."""

import asyncio
import json
import logging
import os
import re
import time
from typing import Dict, List, Optional

import nltk
import numpy as np
import requests
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..constants import CHECKPOINTS
from ..project import save_checkpoint
from ..prompt_templates import get_topic_prompt
from ..progress_tracker import ProgressTracker, SubProgressTracker

# Setup for NLTK
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

logger = logging.getLogger(__name__)


def preprocess_text(text: str) -> str:
    """Preprocess text by tokenizing, removing stopwords, and lemmatizing."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text.lower())
    return " ".join(
        [
            lemmatizer.lemmatize(word)
            for word in tokens
            if word.isalpha() and word not in stop_words
        ]
    )


class TopicAnalyzer:
    """Analyzes transcript segments to identify topics and create segments."""

    def __init__(self, num_topics: int, register: str = "it-workflow", debug: bool = False, progress_tracker: ProgressTracker = None):
        self.num_topics = num_topics
        self.register = register
        self.debug = debug
        self.progress_tracker = progress_tracker
        self.vectorizer = TfidfVectorizer(preprocessor=preprocess_text)
        
        # Enable debug logging if requested
        if self.debug:
            logging.getLogger(__name__).setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled for TopicAnalyzer")

    def _parse_json_response(self, content: str) -> Dict:
        """
        Parse a JSON response from a string with multiple fallback strategies.

        This method attempts to parse a JSON object from a string that may
        contain other text, such as markdown code blocks. It tries several
        strategies to extract and parse the JSON.

        Args:
            content: The string content which is expected to contain a JSON object.

        Returns:
            A dictionary parsed from the JSON, or None if parsing fails.
        """
        # Clean the content
        content = content.strip()
        
        # Strategy 1: Direct JSON parsing
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract JSON from markdown code blocks
        patterns = [
            r"```json\s*({.*?})\s*```",  # ```json {content} ```
            r"```\s*({.*?})\s*```",      # ``` {content} ```
            r"```json\s*\n({.*?})\n```", # ```json\n {content} \n```
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
        
        # Strategy 3: Find JSON object in mixed content
        json_pattern = r'({\s*"[^"]+"\s*:[^}]+})'  # Basic JSON object pattern
        match = re.search(json_pattern, content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Strategy 4: Extract individual fields using regex
        topic_match = re.search(r'"topic"\s*:\s*"([^"]+)"', content)
        keywords_match = re.search(r'"keywords"\s*:\s*\[([^\]]+)\]', content)
        relationship_match = re.search(r'"relationship"\s*:\s*"([^"]+)"', content)
        confidence_match = re.search(r'"confidence"\s*:\s*(\d+)', content)
        
        if topic_match:
            result = {
                "topic": topic_match.group(1),
                "keywords": [],
                "relationship": "NEW",
                "confidence": 50
            }
            
            if keywords_match:
                keywords_str = keywords_match.group(1)
                # Extract keywords from string like "word1", "word2", "word3"
                keywords = re.findall(r'"([^"]+)"', keywords_str)
                result["keywords"] = keywords
            
            if relationship_match:
                result["relationship"] = relationship_match.group(1)
            
            if confidence_match:
                result["confidence"] = int(confidence_match.group(1))
            
            return result
        
        return None
    
    def _validate_response(self, response: Dict) -> Dict:
        """
        Validate and normalize the parsed response from the topic analysis API.

        This ensures that the response dictionary has the required keys with
        valid data types and values, providing defaults where necessary.

        Args:
            response: The parsed dictionary from the API response.

        Returns:
            A validated and normalized dictionary.
        """
        # Ensure required fields exist with defaults
        validated = {
            "topic": response.get("topic", "Uncategorized"),
            "keywords": response.get("keywords", []),
            "relationship": response.get("relationship", "NEW"),
            "confidence": response.get("confidence", 50)
        }
        
        # Validate and normalize topic
        if not validated["topic"] or not isinstance(validated["topic"], str):
            validated["topic"] = "Uncategorized"
        
        # Validate and normalize keywords
        if not isinstance(validated["keywords"], list):
            validated["keywords"] = []
        validated["keywords"] = [str(kw) for kw in validated["keywords"] if kw]
        
        # Validate relationship
        valid_relationships = ["CONTINUATION", "SHIFT", "NEW"]
        if validated["relationship"] not in valid_relationships:
            validated["relationship"] = "NEW"
        
        # Validate confidence
        try:
            confidence = int(validated["confidence"])
            validated["confidence"] = max(0, min(100, confidence))  # Clamp to 0-100
        except (ValueError, TypeError):
            validated["confidence"] = 50
        
        return validated
    
    async def _get_topic_from_openrouter(self, text_chunk: str, max_retries: int = 3) -> Dict:
        """
        Get topic and keywords from the OpenRouter API for a given text chunk.

        This method sends a request to the OpenRouter API to analyze the text
        and returns a structured response with the topic, keywords, and other
        metadata. It includes retry logic with exponential backoff to handle
        transient API errors.

        Args:
            text_chunk: The text to be analyzed.
            max_retries: The maximum number of times to retry the API call.

        Returns:
            A dictionary containing the topic analysis results from the API.
        """
        prompt = get_topic_prompt(self.register, text_chunk)
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set.")

        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    requests.post,
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    data=json.dumps({
                        "model": "microsoft/phi-4",
                        "messages": [{"role": "user", "content": prompt}],
                    }),
                    timeout=30,  # Add timeout
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                
                # Enhanced JSON parsing with multiple fallback strategies
                if self.debug:
                    logger.debug("Raw OpenRouter response (attempt %d/%d):\n%s", attempt + 1, max_retries, content)
                
                parsed_json = self._parse_json_response(content)
                if parsed_json:
                    validated = self._validate_response(parsed_json)
                    if self.debug:
                        logger.debug("Successfully parsed and validated response: %s", validated)
                    return validated
                else:
                    logger.warning("Could not parse JSON from OpenRouter response (attempt %d/%d). Raw content:\n%s", 
                                 attempt + 1, max_retries, content)
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
                        continue
                    return self._validate_response({"topic": "Uncategorized", "keywords": [], "relationship": "NEW", "confidence": 0})

            except requests.RequestException as e:
                logger.error(f"Error calling OpenRouter API (attempt %d/%d): {e}", attempt + 1, max_retries)
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 * (attempt + 1))  # Exponential backoff
                    continue
                return self._validate_response({"topic": "API Error", "keywords": [], "relationship": "NEW", "confidence": 0})
            except (KeyError, json.JSONDecodeError) as e:
                logger.error(f"Error parsing OpenRouter response (attempt %d/%d): {e}", attempt + 1, max_retries)
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))
                    continue
                return self._validate_response({"topic": "Parsing Error", "keywords": [], "relationship": "NEW", "confidence": 0})
        
        # This should never be reached, but just in case
        return self._validate_response({"topic": "Unknown Error", "keywords": [], "relationship": "NEW", "confidence": 0})

    async def analyze_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Analyze each text segment to determine its topic using the OpenRouter API.

        This method asynchronously processes a list of text segments, calling
        the topic analysis API for each one.

        Args:
            segments: A list of dictionaries, where each dictionary represents
                a text segment with a 'content' key.

        Returns:
            The list of segments, updated with topic, keywords, and other
            analysis metadata.
        """
        # Create progress tracker for segment analysis
        segment_descriptions = [f"Segment {i+1}: {seg['content'][:50]}..." for i, seg in enumerate(segments)]
        sub_tracker = None
        if self.progress_tracker:
            sub_tracker = self.progress_tracker.create_sub_progress_tracker("Topic Modeling", segment_descriptions)
        
        # Process segments with progress tracking
        topic_results = []
        for i, seg in enumerate(segments):
            if sub_tracker:
                sub_tracker.update_item_progress(0.0, f"Analyzing segment {i+1}/{len(segments)}")
            
            result = await self._get_topic_from_openrouter(seg["content"])
            topic_results.append(result)
            
            if sub_tracker:
                sub_tracker.complete_item()
        
        # Update segment data with results
        for i, seg in enumerate(segments):
            result = topic_results[i]
            seg["topic"] = result.get("topic", "Uncategorized")
            seg["keywords"] = result.get("keywords", [])
            seg["relationship"] = result.get("relationship", "NEW")
            seg["confidence"] = result.get("confidence", 50)
        
        return segments

    def segment_by_topic(
        self, 
        analyzed_segments: List[Dict], 
        min_segment_duration: float = 30.0,
        max_segment_duration: float = 300.0,
        topic_confidence_threshold: float = 0.7,
        preserve_natural_breaks: bool = True
    ) -> List[Dict]:
        """
        Group consecutive segments that share the same topic into larger segments.

        This method iterates through the analyzed segments and merges adjacent
        segments if their assigned topic is the same, while respecting duration
        constraints and natural content boundaries.

        Args:
            analyzed_segments: A list of segments that have been analyzed for
                their topics.
            min_segment_duration: Minimum duration for merged segments in seconds.
            max_segment_duration: Maximum duration for merged segments in seconds.
            topic_confidence_threshold: Minimum confidence to merge segments.
            preserve_natural_breaks: Whether to respect natural pauses/breaks.

        Returns:
            A new list of segments, where consecutive segments with the same
            topic have been merged according to the specified constraints.
        """
        if not analyzed_segments:
            return []

        final_segments = []
        current_segment = analyzed_segments[0].copy()
        current_segment["content"] = [current_segment["content"]]
        current_segment["segment_id"] = 1
        current_segment["original_segments"] = [analyzed_segments[0]]
        current_segment["merge_info"] = {
            "segments_merged": 1,
            "merge_confidence": current_segment.get("topic_confidence", 1.0)
        }

        for next_seg in analyzed_segments[1:]:
            current_duration = current_segment["end"] - current_segment["start"]
            next_duration = next_seg["end"] - next_seg["start"]
            potential_duration = next_seg["end"] - current_segment["start"]
            
            # Calculate average confidence for merging decision
            current_conf = current_segment.get("topic_confidence", 1.0)
            next_conf = next_seg.get("topic_confidence", 1.0)
            avg_confidence = (current_conf + next_conf) / 2
            
            # Check if we should merge these segments
            should_merge = self._should_merge_segments(
                current_segment, 
                next_seg, 
                potential_duration,
                min_segment_duration,
                max_segment_duration,
                topic_confidence_threshold,
                preserve_natural_breaks
            )
            
            if should_merge:
                # Merge the segments
                current_segment["end"] = next_seg["end"]
                current_segment["content"].append(next_seg["content"])
                current_segment["original_segments"].append(next_seg)
                current_segment["merge_info"]["segments_merged"] += 1
                current_segment["merge_info"]["merge_confidence"] = min(
                    current_segment["merge_info"]["merge_confidence"], 
                    avg_confidence
                )
                
                # Update keywords by merging sets
                if "keywords" in current_segment and "keywords" in next_seg:
                    current_segment["keywords"] = list(set(current_segment["keywords"] + next_seg["keywords"]))
                
                logger.debug(f"Merged segments: {current_segment['topic']} duration={potential_duration:.1f}s")
            else:
                # Finalize current segment and start a new one
                current_segment["content"] = " ".join(current_segment["content"])
                current_segment["duration"] = current_segment["end"] - current_segment["start"]
                final_segments.append(current_segment)
                
                # Start new segment
                current_segment = next_seg.copy()
                current_segment["content"] = [current_segment["content"]]
                current_segment["segment_id"] = len(final_segments) + 1
                current_segment["original_segments"] = [next_seg]
                current_segment["merge_info"] = {
                    "segments_merged": 1,
                    "merge_confidence": next_seg.get("topic_confidence", 1.0)
                }

        # Finalize the last segment
        current_segment["content"] = " ".join(current_segment["content"])
        current_segment["duration"] = current_segment["end"] - current_segment["start"]
        final_segments.append(current_segment)
        
        # Post-process to handle very short segments
        final_segments = self._post_process_segments(final_segments, min_segment_duration)
        
        logger.info(f"Topic segmentation: {len(analyzed_segments)} → {len(final_segments)} segments")
        return final_segments
    
    def _should_merge_segments(
        self,
        current_segment: Dict,
        next_segment: Dict,
        potential_duration: float,
        min_segment_duration: float,
        max_segment_duration: float,
        topic_confidence_threshold: float,
        preserve_natural_breaks: bool
    ) -> bool:
        """
        Determine if two segments should be merged based on various criteria.
        
        Args:
            current_segment: The current segment being built
            next_segment: The next segment to potentially merge
            potential_duration: Duration if segments were merged
            min_segment_duration: Minimum allowed segment duration
            max_segment_duration: Maximum allowed segment duration
            topic_confidence_threshold: Minimum confidence for merging
            preserve_natural_breaks: Whether to respect natural breaks
            
        Returns:
            True if segments should be merged, False otherwise
        """
        # Must have the same topic
        if current_segment["topic"] != next_segment["topic"]:
            return False
        
        # Check duration constraints
        if potential_duration > max_segment_duration:
            return False
        
        # Check confidence threshold
        current_conf = current_segment.get("topic_confidence", 1.0)
        next_conf = next_segment.get("topic_confidence", 1.0)
        if min(current_conf, next_conf) < topic_confidence_threshold:
            return False
        
        # Check for natural breaks if enabled
        if preserve_natural_breaks:
            # Look for natural pause indicators
            current_content = current_segment["content"][-1] if isinstance(current_segment["content"], list) else current_segment["content"]
            next_content = next_segment["content"]
            
            # Check for sentence endings, long pauses, etc.
            if self._has_natural_break(current_content, next_content):
                current_duration = current_segment["end"] - current_segment["start"]
                # Only respect natural breaks if current segment is already reasonably long
                if current_duration >= min_segment_duration:
                    return False
        
        return True
    
    def _has_natural_break(self, current_content: str, next_content: str) -> bool:
        """
        Check if there's a natural break between two content segments.
        
        Args:
            current_content: Content of the current segment
            next_content: Content of the next segment
            
        Returns:
            True if there's a natural break, False otherwise
        """
        if not current_content or not next_content:
            return False
        
        # Check for sentence endings
        sentence_endings = ['. ', '? ', '! ', '.\n', '?\n', '!\n']
        if any(current_content.rstrip().endswith(ending.strip()) for ending in sentence_endings):
            return True
        
        # Check for topic transition phrases
        transition_phrases = [
            'moving on', 'next', 'now let\'s', 'switching to', 'turning to',
            'in conclusion', 'to summarize', 'finally', 'lastly'
        ]
        
        current_lower = current_content.lower()
        next_lower = next_content.lower()
        
        for phrase in transition_phrases:
            if phrase in current_lower or phrase in next_lower:
                return True
        
        return False
    
    def _post_process_segments(self, segments: List[Dict], min_segment_duration: float) -> List[Dict]:
        """
        Post-process segments to handle very short segments.
        
        Args:
            segments: List of segments to process
            min_segment_duration: Minimum segment duration
            
        Returns:
            Processed segments with short segments merged or removed
        """
        if not segments:
            return segments
        
        processed_segments = []
        
        for segment in segments:
            duration = segment.get("duration", segment["end"] - segment["start"])
            
            if duration < min_segment_duration:
                # Try to merge with previous segment if topics are similar
                if (processed_segments and 
                    processed_segments[-1]["topic"] == segment["topic"]):
                    
                    # Merge with previous segment
                    prev_segment = processed_segments[-1]
                    prev_segment["end"] = segment["end"]
                    prev_segment["content"] += " " + segment["content"]
                    prev_segment["duration"] = prev_segment["end"] - prev_segment["start"]
                    prev_segment["original_segments"].extend(segment["original_segments"])
                    prev_segment["merge_info"]["segments_merged"] += segment["merge_info"]["segments_merged"]
                    
                    logger.debug(f"Merged short segment into previous: {segment['topic']} duration={duration:.1f}s")
                    continue
                
                # If can't merge with previous, try to merge with next
                # (This would require looking ahead, skip for now)
                logger.warning(f"Very short segment kept: {segment['topic']} duration={duration:.1f}s")
            
            processed_segments.append(segment)
        
        return processed_segments


def process_transcript(
    transcript: List[Dict], 
    project_path: str, 
    num_topics: int, 
    register: str, 
    debug: bool = False, 
    progress_tracker: ProgressTracker = None,
    min_segment_duration: float = 30.0,
    max_segment_duration: float = 300.0,
    topic_confidence_threshold: float = 0.7,
    preserve_natural_breaks: bool = True
) -> Dict:
    """
    Processes a transcript to model topics and create topic-based segments.
    
    Args:
        transcript: List of transcript segments with timing information
        project_path: Path to the project directory
        num_topics: Number of topics to identify
        register: Analysis register for tailoring the analysis
        debug: Enable debug mode for detailed logging
        progress_tracker: Optional progress tracker
        min_segment_duration: Minimum duration for merged segments in seconds
        max_segment_duration: Maximum duration for merged segments in seconds
        topic_confidence_threshold: Minimum confidence to merge segments
        preserve_natural_breaks: Whether to respect natural pauses/breaks
        
    Returns:
        Dictionary containing topics and merged segments
    """
    if progress_tracker:
        progress_tracker.update_phase_progress(0.0, "Initializing topic analyzer...")
    else:
        print("Starting topic modeling and segmentation...")
    
    analyzer = TopicAnalyzer(num_topics, register, debug, progress_tracker)

    # Analyze segments asynchronously
    if progress_tracker:
        progress_tracker.update_phase_progress(10.0, "Analyzing transcript segments...")
    analyzed_segments = asyncio.run(analyzer.analyze_segments(transcript))

    # Group segments by topic
    if progress_tracker:
        progress_tracker.update_phase_progress(80.0, "Grouping segments by topic...")
    topic_segments = analyzer.segment_by_topic(
        analyzed_segments,
        min_segment_duration=min_segment_duration,
        max_segment_duration=max_segment_duration,
        topic_confidence_threshold=topic_confidence_threshold,
        preserve_natural_breaks=preserve_natural_breaks
    )

    # Create a summary of topics
    topic_summary = {}
    for seg in topic_segments:
        topic = seg["topic"]
        if topic not in topic_summary:
            topic_summary[topic] = {"keywords": set(), "count": 0}
        topic_summary[topic]["keywords"].update(seg["keywords"])
        topic_summary[topic]["count"] += 1

    final_topics = [
        {
            "topic_id": i,
            "dominant_topic": topic,
            "words": sorted(list(details["keywords"])),
            "num_segments": details["count"],
        }
        for i, (topic, details) in enumerate(topic_summary.items())
    ]

    results = {
        "topics": final_topics,
        "segments": topic_segments,
    }

    save_checkpoint(
        project_path,
        CHECKPOINTS["TOPIC_MODELING_COMPLETE"],
        {"results": results},
    )
    if progress_tracker:
        progress_tracker.update_phase_progress(100.0, "Topic modeling and segmentation complete")
    else:
        print("Topic modeling and segmentation complete.")
    return results