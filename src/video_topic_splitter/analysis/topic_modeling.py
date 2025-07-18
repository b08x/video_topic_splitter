"""Topic modeling and segmentation using Large Language Models (LLMs).

This module provides a modular topic analysis system with separated concerns,
better error handling, and cleaner architecture.

The main components are split into separate modules:
- topic_analyzer_config: Configuration management
- response_parser: API response parsing and validation
- async_cache: Async-safe caching implementation
- segment_batcher: Smart batching logic
- topic_analyzer: Core analysis orchestration
"""

import json
import logging
import os
from typing import Dict, List, Optional

import nltk
import tqdm

from ..constants import CHECKPOINTS
from ..project import save_checkpoint
from .topic_analyzer_config import TopicAnalyzerConfig
from .topic_analyzer import TopicAnalyzer
from ..progress_tracker import ProgressTracker

# Setup logger
logger = logging.getLogger(__name__)

# NLTK Data Download
def _ensure_nltk_data():
    """Ensure necessary NLTK data is available."""
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("corpora/stopwords")
    except LookupError:
        logger.info("Downloading NLTK data (punkt, stopwords)...")
        try:
            nltk.download("punkt", quiet=True)
            nltk.download("stopwords", quiet=True)
            logger.info("NLTK data downloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to download NLTK data: {e}")
            raise

_ensure_nltk_data()


def process_transcript(
    transcript_sentences: List[Dict],
    project_path: str,
    num_topics: int = 5,
    register: str = "gen-ai",
    debug: bool = False,
    progress_tracker: Optional[ProgressTracker] = None,
    min_segment_duration: float = 30.0,
    max_segment_duration: float = 300.0,
    topic_confidence_threshold: float = 0.7,
    preserve_natural_breaks: bool = True,
    topic_similarity_threshold: float = 0.6,
    max_merge_passes: int = 3,
) -> Dict:
    """Process a transcript to perform topic modeling and segmentation.

    Uses the modular TopicAnalyzer to identify segments based on topic shifts 
    detected by an LLM. Formats the results and saves them with checkpoints.

    Args:
        transcript_sentences: List of sentence dictionaries with 'start', 'end', 'content'.
        project_path: Path to the project directory for saving results.
        num_topics: Number of topics (kept for compatibility, not directly used).
        register: Analysis register/domain for guiding LLM analysis.
        debug: Enable debug logging.
        progress_tracker: Optional progress tracker for UI updates.
        min_segment_duration: Minimum duration for merged segments in seconds.
        max_segment_duration: Maximum duration for merged segments in seconds.
        topic_confidence_threshold: Minimum confidence to merge segments.
        preserve_natural_breaks: Whether to respect natural pauses/breaks.
        topic_similarity_threshold: Minimum similarity for merging topics.
        max_merge_passes: Maximum number of merge passes.

    Returns:
        Dictionary containing the analysis results with topics and segments.

    Raises:
        ValueError: If TopicAnalyzer initialization fails.
        Exception: If segmentation fails critically.
    """
    logger.info(f"Starting transcript processing for project: {project_path}")
    results_path = os.path.join(project_path, "topic_analysis_results.json")

    # Update progress tracker if provided
    if progress_tracker:
        progress_tracker.update_phase_progress(0.0, "Initializing topic analyzer...")

    # Check for existing results
    if os.path.exists(results_path):
        logger.info(f"Loading existing topic analysis results from: {results_path}")
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            if progress_tracker:
                progress_tracker.update_phase_progress(100.0, "Loaded existing results")
            return results
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load existing results ({e}). Proceeding with analysis.")

    # Initialize analyzer with configuration
    try:
        config = TopicAnalyzerConfig(
            register=register,
            debug=debug,
            min_segment_duration=min_segment_duration,
            max_segment_duration=max_segment_duration,
            topic_confidence_threshold=topic_confidence_threshold,
            preserve_natural_breaks=preserve_natural_breaks,
            topic_similarity_threshold=topic_similarity_threshold,
            max_merge_passes=max_merge_passes,
        )
        analyzer = TopicAnalyzer(config)
        
        if progress_tracker:
            progress_tracker.update_phase_progress(10.0, "Topic analyzer initialized")
            
    except ValueError as ve:
        logger.error(f"Failed to initialize TopicAnalyzer: {ve}")
        raise

    # Identify segments
    logger.info("Identifying topic segments...")
    try:
        if progress_tracker:
            progress_tracker.update_phase_progress(20.0, "Analyzing transcript segments...")
        
        segments = analyzer.identify_segments(transcript_sentences)
        
        if progress_tracker:
            progress_tracker.update_phase_progress(80.0, "Grouping segments by topic...")
            
    except Exception as e:
        logger.error(f"Error during segment identification: {e}", exc_info=True)
        raise

    if not segments:
        logger.warning("No segments were identified. Returning empty results.")
        return {"topics": [], "segments": [], "register": register}

    # Generate metadata and structure results
    logger.info("Formatting analysis results...")
    segment_metadata = []
    topic_summary = {}

    progress_desc = "Generating Segment Metadata"
    iterator = tqdm.tqdm(segments, desc=progress_desc) if not progress_tracker else segments
    
    for i, segment in enumerate(iterator):
        segment_id = i + 1
        start_time = segment.get("start", 0.0)
        end_time = segment.get("end", 0.0)
        topic = segment.get("topic", "Unknown")
        keywords = segment.get("keywords", [])
        content = segment.get("content", "")

        segment_meta = {
            "segment_id": segment_id,
            "start_time": start_time,
            "end_time": end_time,
            "duration": max(0.0, end_time - start_time),
            "dominant_topic": topic,
            "top_keywords": keywords,
            "transcript": content,
        }
        segment_metadata.append(segment_meta)

        # Update topic summary
        if topic != "Unknown":
            if topic not in topic_summary:
                topic_summary[topic] = set()
            topic_summary[topic].update(keywords)

    # Create topics list
    topics_list = [
        {"topic_id": i + 1, "topic": topic_name, "words": sorted(list(kw_set))}
        for i, (topic_name, kw_set) in enumerate(topic_summary.items())
    ]

    # Create final results
    results = {
        "topics": topics_list,
        "segments": segment_metadata,
        "register": register,
        "analysis_info": {
            "total_segments": len(segments),
            "analyzer_config": {
                "register": config.register,
                "batch_size": config.batch_size,
                "similarity_threshold": config.similarity_threshold,
                "model": config.model,
                "debug": config.debug,
            }
        }
    }

    # Save results
    logger.info(f"Saving topic analysis results to: {results_path}")
    try:
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    except (IOError, TypeError) as e:
        logger.error(f"Failed to save results to {results_path}: {e}")

    # Save checkpoint
    logger.info("Saving topic modeling checkpoint...")
    save_checkpoint(
        project_path,
        CHECKPOINTS["TOPIC_MODELING_COMPLETE"],
        {"results_path": results_path, "num_segments": len(segments)}
    )

    if progress_tracker:
        progress_tracker.update_phase_progress(100.0, "Topic modeling and segmentation complete")

    logger.info("Transcript processing and topic modeling complete.")
    return results