"""Smart batching of transcript segments for topic analysis."""

import logging
import numpy as np
from typing import Dict, List
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .topic_analyzer_config import TopicAnalyzerConfig

logger = logging.getLogger(__name__)


class SegmentBatcher:
    """Handles smart batching of transcript segments."""
    
    def __init__(self, config: TopicAnalyzerConfig):
        self.config = config
        self.stop_words = set(stopwords.words("english"))
        self.vectorizer = TfidfVectorizer(
            stop_words=list(self.stop_words),
            max_features=config.tfidf_max_features,
            ngram_range=config.tfidf_ngram_range
        )
    
    def create_batches(self, transcript_sentences: List[Dict]) -> List[List[Dict]]:
        """Create smart batches from transcript sentences."""
        if not transcript_sentences:
            return []
        
        batches = []
        current_batch = []
        
        for i, sentence in enumerate(transcript_sentences):
            current_batch.append(sentence)
            
            should_finalize = False
            if len(current_batch) >= self.config.batch_size:
                # Check for smart boundary if not the last sentence
                if i + 1 < len(transcript_sentences):
                    current_batch_text = " ".join(s["content"] for s in current_batch)
                    next_sentence_text = transcript_sentences[i + 1]["content"]
                    
                    similarity = self._calculate_similarity(current_batch_text, next_sentence_text)
                    logger.debug(f"Similarity between batch ending at {i} and next sentence: {similarity:.3f}")
                    
                    if similarity < self.config.similarity_threshold:
                        logger.debug(f"Low similarity ({similarity:.3f}) detected. Forcing batch break.")
                        should_finalize = True
                    else:
                        should_finalize = True
                else:
                    should_finalize = True
            
            if should_finalize:
                batches.append(current_batch)
                current_batch = []
        
        # Add remaining sentences
        if current_batch:
            batches.append(current_batch)
        
        logger.info(f"Created {len(batches)} batches from {len(transcript_sentences)} sentences.")
        return batches
    
    def combine_batch(self, batch: List[Dict]) -> Dict:
        """Combine sentences within a batch into a single segment."""
        if not batch:
            return {}
        
        return {
            "start": batch[0].get("start", 0.0),
            "end": batch[-1].get("end", 0.0),
            "content": " ".join(s.get("content", "") for s in batch).strip(),
        }
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for TF-IDF similarity calculation."""
        if not text:
            return ""
        
        words = text.lower().split()
        words = [w for w in words if w.isalnum() and w not in self.stop_words]
        return " ".join(words)
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        if not text1 or not text2:
            return 0.0
        
        proc_text1 = self._preprocess_text(text1)
        proc_text2 = self._preprocess_text(text2)
        
        if not proc_text1 or not proc_text2:
            return 0.0
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform([proc_text1, proc_text2])
            
            if tfidf_matrix.shape[0] < 2:
                logger.warning("TF-IDF matrix has fewer than 2 rows, cannot compute similarity.")
                return 0.0
            
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            return float(np.clip(similarity[0][0], 0.0, 1.0))
        
        except ValueError as ve:
            logger.warning(f"ValueError during TF-IDF similarity calculation: {ve}")
            return 0.0
        except Exception as e:
            logger.error(f"Unexpected error calculating similarity: {e}", exc_info=True)
            return 0.0