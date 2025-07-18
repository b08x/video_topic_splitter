"""Configuration for topic analysis."""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

# Default configuration constants
DEFAULT_MODEL = "microsoft/phi-4"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 150
DEFAULT_TIMEOUT = 30
DEFAULT_CACHE_SIZE = 512
DEFAULT_BATCH_SIZE = 5
DEFAULT_MAX_CONCURRENT = 3
DEFAULT_SIMILARITY_THRESHOLD = 0.7
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 5


@dataclass
class TopicAnalyzerConfig:
    """Configuration for the TopicAnalyzer."""
    
    # API Configuration
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout: int = DEFAULT_TIMEOUT
    
    # Processing Configuration
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_delay: int = DEFAULT_RETRY_DELAY
    batch_size: int = DEFAULT_BATCH_SIZE
    max_concurrent: int = DEFAULT_MAX_CONCURRENT
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    
    # Cache Configuration
    cache_size: int = DEFAULT_CACHE_SIZE
    
    # Text Processing Configuration
    max_prev_context_length: int = 500
    max_current_content_length: int = 1500
    tfidf_max_features: int = 1000
    tfidf_ngram_range: Tuple[int, int] = (1, 2)
    
    # Analysis Configuration
    register: str = "gen-ai"
    debug: bool = False  # Add debug support
    
    # Segmentation Configuration (from old implementation)
    min_segment_duration: float = 30.0
    max_segment_duration: float = 300.0
    topic_confidence_threshold: float = 0.7
    preserve_natural_breaks: bool = True
    topic_similarity_threshold: float = 0.6
    max_merge_passes: int = 3
    
    def __post_init__(self):
        """Validate configuration values."""
        self.similarity_threshold = np.clip(self.similarity_threshold, 0.0, 1.0)
        self.topic_similarity_threshold = np.clip(self.topic_similarity_threshold, 0.0, 1.0)
        self.topic_confidence_threshold = np.clip(self.topic_confidence_threshold, 0.0, 1.0)
        
        if self.max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        if self.min_segment_duration < 0:
            raise ValueError("min_segment_duration must be non-negative")
        if self.max_segment_duration < self.min_segment_duration:
            raise ValueError("max_segment_duration must be >= min_segment_duration")