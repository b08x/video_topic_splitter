#!/usr/bin/env python3
"""Topic modeling and transcript segmentation functionality."""

import asyncio
import json
import logging
import os
import re
from typing import Dict, List

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

    def __init__(self, num_topics: int, register: str = "it-workflow"):
        self.num_topics = num_topics
        self.register = register
        self.vectorizer = TfidfVectorizer(preprocessor=preprocess_text)

    async def _get_topic_from_openrouter(self, text_chunk: str) -> Dict:
        """Get topic and keywords from OpenRouter API asynchronously."""
        prompt = get_topic_prompt(self.register, text_chunk)
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set.")

        try:
            response = await asyncio.to_thread(
                requests.post,
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                data=json.dumps({
                    "model": "microsoft/phi-3-medium-128k-instruct",
                    "messages": [{"role": "user", "content": prompt}],
                }),
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            
            try:
                # First, try to parse the content directly as JSON
                return json.loads(content)
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract JSON from a markdown block
                json_match = re.search(r"```json\n({.*?})\n```", content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                else:
                    logger.warning("Could not parse JSON from OpenRouter response. Raw content:\n%s", content)
                    return {"topic": "Uncategorized", "keywords": []}

        except requests.RequestException as e:
            logger.error(f"Error calling OpenRouter API: {e}")
            return {"topic": "Error", "keywords": []}
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing OpenRouter response: {e}")
            return {"topic": "Parsing Error", "keywords": []}

    async def analyze_segments(self, segments: List[Dict]) -> List[Dict]:
        """Analyze each text segment to determine its topic."""
        tasks = [self._get_topic_from_openrouter(seg["content"]) for seg in segments]
        topic_results = await asyncio.gather(*tasks)

        for i, seg in enumerate(segments):
            seg["topic"] = topic_results[i].get("topic", "Uncategorized")
            seg["keywords"] = topic_results[i].get("keywords", [])
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
    transcript: List[Dict], project_path: str, num_topics: int, register: str
) -> Dict:
    """
    Processes a transcript to model topics and create topic-based segments.
    """
    print("Starting topic modeling and segmentation...")
    analyzer = TopicAnalyzer(num_topics, register)

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