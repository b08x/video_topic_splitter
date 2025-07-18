#!/usr/bin/env python3
"""
Enhanced transcript analysis using spaCy for advanced NLP capabilities.

This module provides sophisticated text analysis capabilities using spaCy's
processing pipeline, moving beyond basic string manipulation to linguistically
aware analysis including tokenization, lemmatization, named entity recognition,
and dependency parsing.
"""

import logging
import time
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Global variable to cache the spaCy model
_nlp_model = None


def get_spacy_model():
    """
    Get or load the spaCy model with automatic download if needed.
    
    Returns:
        spaCy language model (nlp object)
    """
    global _nlp_model
    
    if _nlp_model is not None:
        return _nlp_model
    
    try:
        import spacy
        
        # Try to load the medium model first (includes word vectors)
        try:
            _nlp_model = spacy.load("en_core_web_md")
            logger.info("Loaded spaCy model: en_core_web_md")
        except OSError:
            # Fall back to small model if medium is not available
            try:
                _nlp_model = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model: en_core_web_sm")
            except OSError:
                logger.warning("No spaCy English models found. Please install with: python -m spacy download en_core_web_md")
                # Try to download the medium model
                try:
                    spacy.cli.download("en_core_web_md")
                    _nlp_model = spacy.load("en_core_web_md")
                    logger.info("Downloaded and loaded spaCy model: en_core_web_md")
                except Exception as e:
                    logger.error(f"Failed to download spaCy model: {e}")
                    raise RuntimeError("Could not load or download spaCy English model")
        
        return _nlp_model
        
    except ImportError:
        logger.error("spaCy is not installed. Please install with: pip install spacy")
        raise RuntimeError("spaCy is required for enhanced transcript analysis")


class EnhancedTranscriptAnalyzer:
    """
    Advanced transcript analyzer using spaCy for sophisticated NLP analysis.
    
    This class provides comprehensive text analysis capabilities including:
    - Superior tokenization and linguistic processing
    - Lemmatization and stop word filtering
    - Noun chunk extraction for meaningful phrases
    - Named Entity Recognition for structured data extraction
    - Part-of-speech analysis and dependency parsing
    - Semantic similarity capabilities
    """
    
    def __init__(self):
        """Initialize the enhanced transcript analyzer."""
        self.nlp = get_spacy_model()
        self.has_vectors = self.nlp.meta.get("vectors", {}).get("keys", 0) > 0
        
        if not self.has_vectors:
            logger.warning("SpaCy model does not include word vectors. Semantic similarity features will be limited.")
    
    def analyze_transcript_segment(
        self, 
        transcript_segment: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive spaCy-powered analysis of a transcript segment.
        
        Args:
            transcript_segment: List of transcript items with timing and content
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        try:
            if not transcript_segment:
                return {"error": "No transcript data available"}
            
            # Extract and clean text content
            text_content = self._extract_text_content(transcript_segment)
            
            if not text_content.strip():
                return {"error": "No text content found in transcript segment"}
            
            # Process with spaCy pipeline
            doc = self.nlp(text_content)
            
            # Perform comprehensive analysis
            analysis_results = {
                "text_content": text_content,
                "basic_metrics": self._get_basic_metrics(doc, transcript_segment),
                "key_phrases": self._extract_key_phrases(doc),
                "named_entities": self._extract_named_entities(doc),
                "linguistic_features": self._analyze_linguistic_features(doc),
                "actions_and_relationships": self._extract_actions_and_relationships(doc),
                "technical_elements": self._extract_technical_elements(doc),
                "segments_count": len(transcript_segment),
                "transcript_segments": transcript_segment
            }
            
            # Add semantic analysis if vectors are available
            if self.has_vectors:
                analysis_results["semantic_features"] = self._analyze_semantic_features(doc)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in enhanced transcript analysis: {e}")
            return {"error": str(e)}
    
    def _extract_text_content(self, transcript_segment: List[Dict[str, Any]]) -> str:
        """Extract and clean text content from transcript segment."""
        text_parts = []
        
        for item in transcript_segment:
            content = item.get("content") or item.get("text", "")
            if content and content.strip():
                text_parts.append(content.strip())
        
        return " ".join(text_parts)
    
    def _get_basic_metrics(
        self, 
        doc, 
        transcript_segment: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get basic metrics using spaCy's linguistic analysis."""
        # Calculate duration from transcript timing
        total_duration = sum([
            item.get("end", 0) - item.get("start", 0)
            for item in transcript_segment
        ])
        
        # Use spaCy's accurate tokenization and sentence segmentation
        token_count = len([token for token in doc if not token.is_space])
        sentence_count = len(list(doc.sents))
        
        # Calculate speech rate
        speech_rate = token_count / total_duration if total_duration > 0 else 0
        
        return {
            "token_count": token_count,
            "sentence_count": sentence_count,
            "duration": total_duration,
            "speech_rate": speech_rate,
            "average_sentence_length": token_count / sentence_count if sentence_count > 0 else 0
        }
    
    def _extract_key_phrases(self, doc) -> Dict[str, Any]:
        """Extract key phrases using spaCy's noun chunking and lemmatization."""
        # Method 1: Noun chunks (multi-word phrases)
        noun_chunks = [
            chunk.text.lower().strip() 
            for chunk in doc.noun_chunks 
            if len(chunk.text.split()) > 1 and not all(token.is_stop for token in chunk)
        ]
        
        # Method 2: Content lemmas (single words)
        content_lemmas = [
            token.lemma_.lower()
            for token in doc
            if (not token.is_stop and 
                not token.is_punct and 
                not token.is_space and
                token.pos_ in {'NOUN', 'PROPN', 'VERB', 'ADJ'} and
                len(token.text) > 2)
        ]
        
        # Method 3: Technical terms (proper nouns and specialized vocabulary)
        technical_terms = [
            token.text.lower()
            for token in doc
            if (token.pos_ == 'PROPN' or 
                (token.pos_ == 'NOUN' and token.text.isupper()) or
                any(char.isdigit() for char in token.text))
        ]
        
        # Count frequencies
        noun_phrase_freq = Counter(noun_chunks)
        lemma_freq = Counter(content_lemmas)
        technical_freq = Counter(technical_terms)
        
        return {
            "noun_phrases": dict(noun_phrase_freq.most_common(10)),
            "key_lemmas": dict(lemma_freq.most_common(15)),
            "technical_terms": dict(technical_freq.most_common(10)),
            "phrase_extraction_method": "spacy_noun_chunks_and_lemmatization"
        }
    
    def _extract_named_entities(self, doc) -> Dict[str, Any]:
        """Extract and categorize named entities."""
        entities = defaultdict(list)
        
        # Standard spaCy entity labels we're interested in
        target_labels = {
            'PERSON', 'ORG', 'GPE', 'DATE', 'TIME', 'MONEY', 'PERCENT',
            'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'NORP'
        }
        
        for ent in doc.ents:
            if ent.label_ in target_labels:
                entities[ent.label_].append({
                    "text": ent.text,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                    "confidence": getattr(ent, 'confidence', 0.0)
                })
        
        # Remove duplicates while preserving order
        for label in entities:
            seen = set()
            unique_entities = []
            for entity in entities[label]:
                entity_text = entity["text"].lower()
                if entity_text not in seen:
                    seen.add(entity_text)
                    unique_entities.append(entity)
            entities[label] = unique_entities
        
        # Add summary statistics
        entity_summary = {
            "total_entities": sum(len(ents) for ents in entities.values()),
            "entity_types": len(entities),
            "most_mentioned_type": max(entities.keys(), key=lambda x: len(entities[x])) if entities else None
        }
        
        return {
            "entities": dict(entities),
            "summary": entity_summary,
            "extraction_method": "spacy_ner"
        }
    
    def _analyze_linguistic_features(self, doc) -> Dict[str, Any]:
        """Analyze linguistic features using spaCy's POS tagging and parsing."""
        # Part-of-speech distribution
        pos_counts = Counter(token.pos_ for token in doc if not token.is_space)
        
        # Dependency relations (grammatical relationships)
        dep_counts = Counter(token.dep_ for token in doc if not token.is_space)
        
        # Morphological features
        morphology = defaultdict(Counter)
        for token in doc:
            if token.morph:
                for feature in token.morph:
                    key, value = feature.split('=') if '=' in feature else (feature, 'True')
                    morphology[key][value] += 1
        
        return {
            "part_of_speech_distribution": dict(pos_counts),
            "dependency_relations": dict(dep_counts.most_common(10)),
            "morphological_features": {k: dict(v) for k, v in morphology.items()},
            "analysis_method": "spacy_linguistic_analysis"
        }
    
    def _extract_actions_and_relationships(self, doc) -> Dict[str, Any]:
        """Extract subject-verb-object relationships and key actions."""
        svo_triplets = []
        key_actions = []
        
        for token in doc:
            # Find main verbs (not auxiliary)
            if token.pos_ == 'VERB' and token.dep_ in {'ROOT', 'conj'} and not token.text.lower() in {'be', 'have', 'do'}:
                # Find subjects
                subjects = [
                    child.text for child in token.children 
                    if child.dep_ in {'nsubj', 'nsubjpass'}
                ]
                
                # Find objects
                objects = [
                    child.text for child in token.children 
                    if child.dep_ in {'dobj', 'pobj', 'attr'}
                ]
                
                # Create SVO triplet
                if subjects or objects:
                    triplet = {
                        "verb": token.lemma_,
                        "verb_text": token.text,
                        "subjects": subjects,
                        "objects": objects,
                        "full_phrase": self._get_verb_phrase(token)
                    }
                    svo_triplets.append(triplet)
                
                # Track key actions
                key_actions.append({
                    "action": token.lemma_,
                    "text": token.text,
                    "tense": self._get_tense(token),
                    "phrase": self._get_verb_phrase(token)
                })
        
        return {
            "subject_verb_object_triplets": svo_triplets[:10],
            "key_actions": key_actions[:15],
            "total_actions": len(key_actions),
            "extraction_method": "spacy_dependency_parsing"
        }
    
    def _extract_technical_elements(self, doc) -> List[str]:
        """Extract technical elements and specialized vocabulary."""
        technical_elements = set()
        
        # Technical patterns
        for token in doc:
            text = token.text.lower()
            
            # Software/technology related terms
            if any(term in text for term in ['api', 'sdk', 'app', 'software', 'system', 'code', 'data', 'server', 'client']):
                technical_elements.add(token.text)
            
            # Version numbers, IDs, technical specifications
            if any(char.isdigit() for char in text) and len(text) > 2:
                technical_elements.add(token.text)
            
            # Acronyms (all caps, 2+ letters)
            if token.text.isupper() and len(token.text) >= 2 and token.pos_ in {'NOUN', 'PROPN'}:
                technical_elements.add(token.text)
        
        # Technical noun phrases
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower()
            if any(tech_word in chunk_text for tech_word in [
                'interface', 'protocol', 'framework', 'library', 'module', 
                'component', 'service', 'platform', 'database', 'network'
            ]):
                technical_elements.add(chunk.text)
        
        return list(technical_elements)
    
    def _analyze_semantic_features(self, doc) -> Dict[str, Any]:
        """Analyze semantic features using word vectors (if available)."""
        if not self.has_vectors:
            return {"error": "Word vectors not available in current spaCy model"}
        
        # Get sentence vectors for semantic analysis
        sentences = list(doc.sents)
        sentence_vectors = [sent.vector for sent in sentences if sent.vector_norm > 0]
        
        # Calculate document coherence (average pairwise similarity)
        coherence_scores = []
        if len(sentence_vectors) > 1:
            from scipy.spatial.distance import cosine
            for i in range(len(sentence_vectors)):
                for j in range(i + 1, len(sentence_vectors)):
                    similarity = 1 - cosine(sentence_vectors[i], sentence_vectors[j])
                    coherence_scores.append(similarity)
        
        avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0
        
        return {
            "document_coherence": avg_coherence,
            "sentence_count_with_vectors": len(sentence_vectors),
            "semantic_analysis_available": True,
            "vector_dimension": doc.vector.shape[0] if doc.vector.shape else 0
        }
    
    def _get_verb_phrase(self, verb_token) -> str:
        """Extract the full verb phrase including auxiliaries and particles."""
        phrase_tokens = [verb_token]
        
        # Add auxiliaries and particles
        for child in verb_token.children:
            if child.dep_ in {'aux', 'auxpass', 'neg', 'prt'}:
                phrase_tokens.append(child)
        
        # Sort by position in text
        phrase_tokens.sort(key=lambda x: x.i)
        return " ".join(token.text for token in phrase_tokens)
    
    def _get_tense(self, verb_token) -> str:
        """Determine the tense of a verb token."""
        if verb_token.morph:
            for feature in verb_token.morph:
                if feature.startswith('Tense='):
                    return feature.split('=')[1].lower()
        
        # Fallback to simple heuristics
        if verb_token.tag_.startswith('VB'):
            if verb_token.tag_ in {'VBD', 'VBN'}:
                return 'past'
            elif verb_token.tag_ in {'VBG'}:
                return 'present'
            elif verb_token.tag_ in {'VBZ', 'VBP'}:
                return 'present'
        
        return 'unknown'

    def find_similar_segments(
        self, 
        query_text: str, 
        segments: List[str], 
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find segments most similar to a query using semantic similarity.
        
        Args:
            query_text: Text to search for
            segments: List of text segments to search in
            top_k: Number of top results to return
            
        Returns:
            List of (segment, similarity_score) tuples
        """
        if not self.has_vectors:
            logger.warning("Semantic similarity requires word vectors. Please install en_core_web_md model.")
            return []
        
        try:
            query_doc = self.nlp(query_text)
            similarities = []
            
            for segment in segments:
                segment_doc = self.nlp(segment)
                similarity = query_doc.similarity(segment_doc)
                similarities.append((segment, similarity))
            
            # Sort by similarity score (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error in semantic similarity search: {e}")
            return []


def analyze_transcript_with_spacy(
    transcript_segment: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Convenience function for enhanced transcript analysis.
    
    Args:
        transcript_segment: List of transcript items with timing and content
        
    Returns:
        Dictionary containing comprehensive spaCy-powered analysis
    """
    analyzer = EnhancedTranscriptAnalyzer()
    return analyzer.analyze_transcript_segment(transcript_segment)