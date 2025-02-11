"""Topic modeling using OpenRouter with microsoft/phi-4 model."""

import os
import json
import time
import asyncio
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
import progressbar
from openai import OpenAI, AsyncOpenAI
import numpy as np
from collections import deque
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import tqdm

from .constants import CHECKPOINTS
from .project import save_checkpoint
from .prompt_templates import get_topic_prompt

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


class TopicAnalyzer:
    """Class to handle topic analysis using OpenRouter's phi-4 model."""
    
    def __init__(self, max_retries: int = 3, retry_delay: int = 5, 
                 batch_size: int = 5, max_concurrent: int = 3,
                 similarity_threshold: float = 0.7, register: str = "gen-ai"):
        """Initialize the TopicAnalyzer.
        
        Args:
            max_retries: Maximum number of retries for failed API calls
            retry_delay: Delay between retries in seconds
            batch_size: Number of sentences to analyze together
            max_concurrent: Maximum number of concurrent API calls
            similarity_threshold: Threshold for text similarity (0-1)
            register: Analysis register (it-workflow, gen-ai, tech-support)
        """
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")
            
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.async_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.similarity_threshold = similarity_threshold
        self.register = register
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Initialize text processing tools
        self.vectorizer = TfidfVectorizer(
            stop_words=stopwords.words('english'),
            max_features=1000
        )
        self.stop_words = set(stopwords.words('english'))

    @lru_cache(maxsize=1000)
    def _get_cached_analysis(self, content: str, prev_content: Optional[str] = None) -> Optional[Dict]:
        """Get cached analysis result for a content string."""
        return None  # Cache miss by default, actual caching handled by lru_cache decorator

    async def analyze_segment_async(self, current_segment: Dict, previous_segment: Optional[Dict] = None) -> Dict:
        """Analyze a segment considering previous context and prosodic features asynchronously."""
        async with self.semaphore:  # Limit concurrent API calls
            # Check cache first
            cache_key = current_segment['content']
            prev_key = previous_segment['content'] if previous_segment else None
            cached_result = self._get_cached_analysis(cache_key, prev_key)
            if cached_result:
                return cached_result

            context = ""
            if previous_segment:
                context = (f"Previous segment context:\n"
                          f"Content: {previous_segment['content']}\n"
                          f"Topic: {previous_segment.get('topic', 'Unknown')}\n")
                
                # Add prosodic context if available
                if 'prosodic_features' in previous_segment:
                    pf = previous_segment['prosodic_features']
                    context += (f"Previous prosodic features:\n"
                              f"Pitch mean: {pf['pitch']['mean']}\n"
                              f"Loudness mean: {pf['loudness']['mean']}\n"
                              f"Rhythm BPM: {pf['rhythm']['bpm']}\n\n")
            
            context += f"Analyze the following segment:\n{current_segment['content']}"
            
            # Add current prosodic features if available
            if 'prosodic_features' in current_segment:
                pf = current_segment['prosodic_features']
                context += (f"\n\nProsodic features:\n"
                          f"Pitch mean: {pf['pitch']['mean']}\n"
                          f"Loudness mean: {pf['loudness']['mean']}\n"
                          f"Rhythm BPM: {pf['rhythm']['bpm']}")
            
            prompt = get_topic_prompt(self.register, context)

            for attempt in range(self.max_retries):
                try:
                    completion = await self.async_client.chat.completions.create(
                        extra_headers={
                            "HTTP-Referer": "https://video-topic-splitter.example",
                            "X-Title": "Video Topic Splitter",
                        },
                        model="microsoft/phi-4",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                    )
                    
                    response_text = completion.choices[0].message.content
                    try:
                        # Extract JSON from response (handle potential text wrapping)
                        json_str = response_text[response_text.find("{"):response_text.rfind("}")+1]
                        result = json.loads(json_str)
                        return result
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON response: {e}")
                        print(f"Raw response: {response_text}")
                        # Return a formatted response even if JSON parsing fails
                        return {
                            "topic": "Unknown",
                            "keywords": [],
                            "relationship": "NEW",
                            "confidence": 0
                        }

                except Exception as e:
                    if attempt < self.max_retries - 1:
                        print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {self.retry_delay} seconds...")
                        await asyncio.sleep(self.retry_delay)
                    else:
                        print(f"All attempts failed: {str(e)}")
                        raise

    def analyze_segment(self, current_segment: Dict, previous_segment: Optional[Dict] = None) -> Dict:
        """Synchronous wrapper for analyze_segment_async."""
        return asyncio.run(self.analyze_segment_async(current_segment, previous_segment))

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis."""
        # Tokenize and remove stopwords
        words = text.lower().split()
        words = [w for w in words if w not in self.stop_words]
        return " ".join(words)

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two text segments."""
        if not text1 or not text2:
            return 0.0
        
        # Preprocess texts
        text1 = self._preprocess_text(text1)
        text2 = self._preprocess_text(text2)
        
        # Calculate TF-IDF and similarity
        tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return float(similarity)

    def _calculate_prosodic_similarity(self, segment1: Dict, segment2: Dict) -> Optional[float]:
        """Calculate similarity between prosodic features of two segments."""
        if not ('prosodic_features' in segment1 and 'prosodic_features' in segment2):
            return None
            
        pf1 = segment1['prosodic_features']
        pf2 = segment2['prosodic_features']
        
        # Calculate normalized differences for each feature
        pitch_diff = abs(pf1['pitch']['mean'] - pf2['pitch']['mean']) / max(pf1['pitch']['mean'], pf2['pitch']['mean'])
        loudness_diff = abs(pf1['loudness']['mean'] - pf2['loudness']['mean']) / max(pf1['loudness']['mean'], pf2['loudness']['mean'])
        rhythm_diff = abs(pf1['rhythm']['bpm'] - pf2['rhythm']['bpm']) / max(pf1['rhythm']['bpm'], pf2['rhythm']['bpm'])
        
        # Average the differences and convert to similarity score
        similarity = 1 - ((pitch_diff + loudness_diff + rhythm_diff) / 3)
        return float(similarity)

    def _create_batches(self, transcript: List[Dict]) -> List[List[Dict]]:
        """Create batches of sentences for analysis with smart boundary detection."""
        batches = []
        current_batch = []
        
        for i, sentence in enumerate(transcript):
            current_batch.append(sentence)
            
            if len(current_batch) >= self.batch_size:
                # Check if this is a good boundary point
                if i + 1 < len(transcript):
                    current_text = " ".join(s["content"] for s in current_batch)
                    next_text = transcript[i + 1]["content"]
                    text_similarity = self._calculate_similarity(current_text, next_text)
                    
                    # Calculate prosodic similarity if available
                    prosodic_similarity = self._calculate_prosodic_similarity(
                        current_batch[-1], transcript[i + 1]
                    )
                    
                    # Combine text and prosodic similarity if both available
                    if prosodic_similarity is not None:
                        similarity = (text_similarity + prosodic_similarity) / 2
                    else:
                        similarity = text_similarity
                    
                    # If similarity is low, this might be a natural boundary
                    if similarity < self.similarity_threshold:
                        batches.append(current_batch)
                        current_batch = []
                        continue
                
                # If we didn't find a natural boundary, use regular batching
                batches.append(current_batch)
                current_batch = []
        
        if current_batch:  # Add remaining sentences
            batches.append(current_batch)
            
        return batches

    def _combine_batch(self, batch: List[Dict]) -> Dict:
        """Combine a batch of sentences into a single segment."""
        combined = {
            "start": batch[0]["start"],
            "end": batch[-1]["end"],
            "content": " ".join(s["content"] for s in batch)
        }
        
        # Combine prosodic features if available
        if all('prosodic_features' in s for s in batch):
            pitch_means = [s['prosodic_features']['pitch']['mean'] for s in batch]
            loudness_means = [s['prosodic_features']['loudness']['mean'] for s in batch]
            bpms = [s['prosodic_features']['rhythm']['bpm'] for s in batch]
            
            combined['prosodic_features'] = {
                'pitch': {'mean': np.mean(pitch_means)},
                'loudness': {'mean': np.mean(loudness_means)},
                'rhythm': {'bpm': np.mean(bpms)}
            }
            
        return combined

    async def _analyze_batch(self, batch: Dict, previous_batch: Optional[Dict] = None) -> Dict:
        """Analyze a batch of sentences."""
        return await self.analyze_segment_async(batch, previous_batch)

    async def _analyze_batches(self, batches: List[List[Dict]]) -> List[Dict]:
        """Analyze multiple batches concurrently with progress tracking."""
        analyses = []
        previous_batch = None
        total_batches = len(batches)
        
        with tqdm.tqdm(total=total_batches, desc="Analyzing batches") as pbar:
            for i in range(0, len(batches), self.max_concurrent):
                current_batches = batches[i:i + self.max_concurrent]
                combined_batches = [self._combine_batch(batch) for batch in current_batches]
                
                tasks = []
                for batch in combined_batches:
                    task = self._analyze_batch(batch, previous_batch)
                    tasks.append(task)
                
                batch_results = await asyncio.gather(*tasks)
                analyses.extend(batch_results)
                
                if combined_batches:
                    previous_batch = combined_batches[-1]
                
                pbar.update(len(current_batches))
        
        return analyses

    def identify_segments(self, transcript: List[Dict]) -> List[Dict]:
        """Identify segments based on topic analysis using batched and parallel processing."""
        print("Identifying segments based on topics and prosodic features...")
        
        # Create batches of sentences with smart boundary detection
        print("Creating smart batches...")
        batches = self._create_batches(transcript)
        print(f"Created {len(batches)} optimized batches for analysis...")
        
        # Analyze batches with progress tracking
        analyses = asyncio.run(self._analyze_batches(batches))
        
        # Process results into segments
        segments = []
        current_segment = None
        previous_segment = None
        
        for i, (batch, analysis) in enumerate(zip(batches, analyses)):
            if not current_segment:
                current_segment = self._combine_batch(batch)
                current_segment["topic"] = analysis["topic"]
                current_segment["keywords"] = analysis["keywords"]
            else:
                # Calculate prosodic change if features available
                prosodic_change = False
                if ('prosodic_features' in current_segment and 
                    'prosodic_features' in batch[0]):
                    similarity = self._calculate_prosodic_similarity(
                        current_segment, batch[0]
                    )
                    if similarity is not None and similarity < self.similarity_threshold:
                        prosodic_change = True
                
                # Decide whether to start a new segment
                if ((analysis["relationship"] == "NEW" and analysis["confidence"] > 70) or
                    (analysis["relationship"] == "SHIFT" and analysis["confidence"] > 85) or
                    prosodic_change):
                    # Finalize current segment
                    segments.append(current_segment)
                    
                    # Start new segment
                    previous_segment = current_segment
                    current_segment = self._combine_batch(batch)
                    current_segment["topic"] = analysis["topic"]
                    current_segment["keywords"] = analysis["keywords"]
                else:
                    # Continue current segment
                    current_segment["end"] = batch[-1]["end"]
                    current_segment["content"] += " " + " ".join(s["content"] for s in batch)
                    
                    # Update prosodic features if available
                    if ('prosodic_features' in current_segment and 
                        'prosodic_features' in batch[-1]):
                        current_pf = current_segment['prosodic_features']
                        batch_pf = batch[-1]['prosodic_features']
                        current_segment['prosodic_features'] = {
                            'pitch': {
                                'mean': (current_pf['pitch']['mean'] + batch_pf['pitch']['mean']) / 2
                            },
                            'loudness': {
                                'mean': (current_pf['loudness']['mean'] + batch_pf['loudness']['mean']) / 2
                            },
                            'rhythm': {
                                'bpm': (current_pf['rhythm']['bpm'] + batch_pf['rhythm']['bpm']) / 2
                            }
                        }
        
        # Handle last segment
        if current_segment:
            segments.append(current_segment)

        print(f"Identified {len(segments)} segments.")
        return segments

def process_transcript(transcript: List[Dict], project_path: str, num_topics: int = 5, register: str = "gen-ai") -> Dict:
    """Process transcript for topic modeling and segmentation."""
    analyzer = TopicAnalyzer(register=register)
    
    # Identify segments with topic analysis
    segments = analyzer.identify_segments(transcript)
    
    # Generate metadata
    metadata = []
    for i, segment in enumerate(progressbar.progressbar(segments)):
        segment_metadata = {
            "segment_id": i + 1,
            "start_time": segment["start"],
            "end_time": segment["end"],
            "duration": segment["end"] - segment["start"],
            "dominant_topic": segment["topic"],
            "top_keywords": segment["keywords"],
            "transcript": segment["content"],
        }
        
        # Include prosodic features if available
        if 'prosodic_features' in segment:
            segment_metadata['prosodic_features'] = segment['prosodic_features']
            
        metadata.append(segment_metadata)

    # Create results structure
    results = {
        "topics": [
            {
                "topic_id": i,
                "topic": segment["topic"],
                "words": segment["keywords"]
            }
            for i, segment in enumerate(segments)
        ],
        "segments": metadata,
        "register": register
    }

    # Save results
    results_path = os.path.join(project_path, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

    save_checkpoint(project_path, CHECKPOINTS['TOPIC_MODELING_COMPLETE'], {
        'results': results
    })

    return results
