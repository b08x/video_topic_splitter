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

    def __init__(self, num_topics: int, register: str = "it-workflow", debug: bool = False):
        self.num_topics = num_topics
        self.register = register
        self.debug = debug
        self.vectorizer = TfidfVectorizer(preprocessor=preprocess_text)
        
        # Enable debug logging if requested
        if self.debug:
            logging.getLogger(__name__).setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled for TopicAnalyzer")

    def _parse_json_response(self, content: str) -> Dict:
        """Parse JSON from OpenRouter response with multiple fallback strategies."""
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
        """Validate and normalize the response from OpenRouter."""
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
        """Get topic and keywords from OpenRouter API with retry logic."""
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
        """Analyze each text segment to determine its topic."""
        tasks = [self._get_topic_from_openrouter(seg["content"]) for seg in segments]
        topic_results = await asyncio.gather(*tasks)

        for i, seg in enumerate(segments):
            result = topic_results[i]
            seg["topic"] = result.get("topic", "Uncategorized")
            seg["keywords"] = result.get("keywords", [])
            seg["relationship"] = result.get("relationship", "NEW")
            seg["confidence"] = result.get("confidence", 50)
        return segments

    def segment_by_topic(self, analyzed_segments: List[Dict]) -> List[Dict]:
        """Group continuous segments with the same topic."""
        if not analyzed_segments:
            return []

        final_segments = []
        current_segment = analyzed_segments[0].copy()
        current_segment["content"] = [current_segment["content"]]
        current_segment["segment_id"] = 1

        for next_seg in analyzed_segments[1:]:
            if next_seg["topic"] == current_segment["topic"]:
                current_segment["end"] = next_seg["end"]
                current_segment["content"].append(next_seg["content"])
            else:
                current_segment["content"] = " ".join(current_segment["content"])
                final_segments.append(current_segment)
                current_segment = next_seg.copy()
                current_segment["content"] = [current_segment["content"]]
                current_segment["segment_id"] = len(final_segments) + 1

        current_segment["content"] = " ".join(current_segment["content"])
        final_segments.append(current_segment)
        return final_segments


def process_transcript(
    transcript: List[Dict], project_path: str, num_topics: int, register: str, debug: bool = False
) -> Dict:
    """
    Processes a transcript to model topics and create topic-based segments.
    """
    print("Starting topic modeling and segmentation...")
    analyzer = TopicAnalyzer(num_topics, register, debug)

    # Analyze segments asynchronously
    analyzed_segments = asyncio.run(analyzer.analyze_segments(transcript))

    # Group segments by topic
    topic_segments = analyzer.segment_by_topic(analyzed_segments)

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
    print("Topic modeling and segmentation complete.")
    return results