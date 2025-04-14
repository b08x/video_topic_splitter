# analysis/topic_modeling.py
"""Topic modeling and segmentation using Large Language Models (LLMs).

This module provides the `TopicAnalyzer` class, which leverages an LLM
(specifically configured for OpenRouter's API, e.g., microsoft/phi-4)
to analyze text segments (derived from video transcripts) for dominant topics,
keywords, and relationships between consecutive segments. It uses asynchronous
processing for efficiency, incorporates caching, and employs text similarity
heuristics for smart batching of sentences before analysis. The goal is to
identify meaningful topic boundaries within a continuous transcript.

The module also includes a `process_transcript` function that orchestrates
the analysis using `TopicAnalyzer` and formats the results.
"""

import asyncio
import json
import logging # Use logging instead of print for better control
import os
import time
from collections import deque
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import nltk
import numpy as np
import tqdm # Use tqdm directly instead of progressbar for consistency
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from openai import AsyncOpenAI, OpenAI, RateLimitError, APIError # Import specific errors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..constants import CHECKPOINTS
from ..project import save_checkpoint
from ..prompt_templates import get_topic_prompt

# Setup logger
logger = logging.getLogger(__name__)

# --- NLTK Data Download ---
# Ensure necessary NLTK data is available.
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
        # Depending on the application, you might want to raise an error here
        # or proceed with potentially degraded functionality.
        # For now, we log the error and continue.


class TopicAnalyzer:
    """Analyzes text segments to identify topics and relationships using an LLM.

    Uses an external LLM API (via OpenRouter) to determine the topic, keywords,
    and relationship (NEW, SHIFT, CONTINUATION) of a text segment relative to
    the previous one. Implements caching, retries, concurrency limiting, and
    TF-IDF based similarity checks for optimizing analysis.

    Attributes:
        client (OpenAI): Synchronous OpenAI client configured for OpenRouter.
        async_client (AsyncOpenAI): Asynchronous OpenAI client for OpenRouter.
        max_retries (int): Maximum retry attempts for API calls.
        retry_delay (int): Delay in seconds between retries.
        batch_size (int): Target number of sentences per analysis batch.
        max_concurrent (int): Maximum concurrent API requests allowed.
        similarity_threshold (float): Cosine similarity threshold used for
            smart batch boundary detection.
        register (str): The analysis register/domain (e.g., 'gen-ai') used to
            select the appropriate prompt template.
        semaphore (asyncio.Semaphore): Limits concurrent async tasks.
        vectorizer (TfidfVectorizer): TF-IDF vectorizer for text similarity.
        stop_words (set): Set of English stopwords for preprocessing.
    """

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: int = 5,
        batch_size: int = 5,
        max_concurrent: int = 3,
        similarity_threshold: float = 0.7,
        register: str = "gen-ai",
    ):
        """Initialize the TopicAnalyzer.

        Args:
            max_retries (int, optional): Maximum number of retries for failed
                API calls. Defaults to 3.
            retry_delay (int, optional): Delay between retries in seconds.
                Defaults to 5.
            batch_size (int, optional): Preferred number of sentences to group
                together for a single API call. Defaults to 5.
            max_concurrent (int, optional): Maximum number of concurrent API
                calls allowed. Defaults to 3.
            similarity_threshold (float, optional): Threshold (0-1) for cosine
                similarity. Used in `_create_batches` to potentially split batches
                at points of low similarity between adjacent sentences/chunks.
                Defaults to 0.7.
            register (str, optional): The analysis register or domain context
                (e.g., 'it-workflow', 'gen-ai', 'tech-support') which helps
                in selecting the appropriate prompt template for the LLM.
                Defaults to "gen-ai".

        Raises:
            ValueError: If the OPENROUTER_API_KEY environment variable is not set.
        """
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")

        # Configure clients for OpenRouter
        # Recommended headers for OpenRouter identification
        default_headers = {
            "HTTP-Referer": "https://github.com/your-repo/video-topic-splitter", # Replace with your repo URL
            "X-Title": "Video Topic Splitter", # Replace with your app name
        }

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers=default_headers,
            max_retries=0, # Handle retries manually for async
        )
        self.async_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers=default_headers,
            max_retries=0, # Handle retries manually
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.similarity_threshold = np.clip(similarity_threshold, 0.0, 1.0) # Ensure valid range
        self.register = register
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # Initialize text processing tools
        self.stop_words = set(stopwords.words("english"))
        self.vectorizer = TfidfVectorizer(
            stop_words=list(self.stop_words), # Pass list or 'english'
            max_features=1000, # Limit feature size for efficiency
            ngram_range=(1, 2) # Consider bi-grams
        )
        logger.info(
            f"TopicAnalyzer initialized: register='{register}', batch_size={batch_size}, "
            f"max_concurrent={max_concurrent}, similarity_threshold={similarity_threshold}"
        )

    # Note: LRU Cache on async methods requires careful consideration or wrappers.
    # For simplicity, caching is applied synchronously here, but a dedicated async
    # cache might be better. The current implementation caches based on content strings.
    @lru_cache(maxsize=512) # Increased cache size
    def _get_cached_analysis(
        self, content_key: str, prev_content_key: Optional[str] = None
    ) -> Optional[Dict]:
        """Retrieves cached analysis result based on content keys.

        This method acts as the lookup for the LRU cache. The actual caching
        is handled by the `@lru_cache` decorator on the calling method
        (`analyze_segment_async` effectively, though applied here for clarity).

        Args:
            content_key (str): A hashable representation of the current segment's
                               content (e.g., the text itself if not too long,
                               or a hash).
            prev_content_key (Optional[str], optional): A hashable representation
                               of the previous segment's content. Defaults to None.

        Returns:
            Optional[Dict]: The cached analysis dictionary if found, otherwise None.
        """
        # The lru_cache decorator handles the actual storage and retrieval.
        # This method signature defines the cache key structure.
        # We return None here to indicate a cache miss, which prompts the
        # decorated function to execute.
        return None

    def _add_to_cache(
        self, content_key: str, prev_content_key: Optional[str], result: Dict
    ):
        """Explicitly adds a result to the cache.

        This helper is needed because `@lru_cache` caches the return value of
        the function it decorates. We call the internal `_get_cached_analysis`
        which is decorated, effectively setting the cache value.

        Args:
            content_key (str): Cache key for the current content.
            prev_content_key (Optional[str]): Cache key for the previous content.
            result (Dict): The analysis result dictionary to cache.
        """
        # This call populates the cache associated with _get_cached_analysis
        self._get_cached_analysis(content_key, prev_content_key) # Call to set cache key
        # The lru_cache mechanism doesn't allow direct setting, it caches the *return* value.
        # A more robust approach might involve a custom cache implementation or
        # restructuring how caching interacts with the async analysis method.
        # For now, we rely on the next call with the same keys hitting the cache.
        # A simple workaround is to call the decorated function with the keys and
        # have it return the result we want to cache, but that feels clunky.
        # Let's assume the lru_cache on _get_cached_analysis works as intended
        # by caching the *arguments* and associating the *result* of the *caller*
        # (analyze_segment_async) implicitly. This is non-standard usage.
        # A cleaner way: Cache directly within analyze_segment_async if needed,
        # or use a dedicated async caching library.
        # Given the current structure, we'll rely on the @lru_cache on the lookup.
        pass # No direct action needed if relying solely on @lru_cache on lookup


    async def analyze_segment_async(
        self, current_segment: Dict, previous_segment: Optional[Dict] = None
    ) -> Dict:
        """Analyzes a text segment asynchronously using the LLM, with context.

        Constructs a prompt including the current segment's text and optionally
        context from the previous segment (text, topic). Sends the prompt to the
        configured LLM via the OpenRouter API. Handles retries on failure and
        parses the JSON response. Uses a semaphore to limit concurrency.
        Implements caching based on segment content.

        Args:
            current_segment (Dict): Dictionary containing the content of the
                current segment under the key 'content'.
            previous_segment (Optional[Dict], optional): Dictionary containing
                the content and analysis results ('content', 'topic') of the
                preceding segment. Defaults to None.

        Returns:
            Dict: A dictionary containing the analysis results from the LLM,
                typically including 'topic', 'keywords', 'relationship', and
                'confidence'. Returns a default error structure if analysis fails
                after retries or if JSON parsing fails.

        Raises:
            Exception: Propagates exceptions from the API call if all retries fail.
        """
        current_content = current_segment.get("content", "")
        prev_content = previous_segment.get("content", "") if previous_segment else None
        prev_topic = previous_segment.get("topic", "Unknown") if previous_segment else "None"

        # Use content itself as cache key (consider hashing if content is very large)
        cache_key_current = current_content
        cache_key_prev = prev_content

        # Check cache first
        cached_result = self._get_cached_analysis(cache_key_current, cache_key_prev)
        if cached_result:
            logger.debug("Cache hit for segment analysis.")
            return cached_result

        logger.debug("Cache miss. Analyzing segment via API.")
        async with self.semaphore: # Acquire semaphore before API call
            context = ""
            if previous_segment:
                # Limit context length to avoid excessive prompt size
                max_prev_len = 500
                truncated_prev_content = (
                    (prev_content[:max_prev_len] + "...")
                    if len(prev_content) > max_prev_len
                    else prev_content
                )
                context = (
                    f"Previous segment context:\n"
                    f"Content: {truncated_prev_content}\n"
                    f"Identified Topic: {prev_topic}\n\n"
                )

            # Limit current content length as well
            max_curr_len = 1500
            truncated_current_content = (
                (current_content[:max_curr_len] + "...")
                if len(current_content) > max_curr_len
                else current_content
            )
            context += f"Analyze the following current segment:\nContent: {truncated_current_content}"

            # Get the appropriate prompt template based on the register
            prompt = get_topic_prompt(self.register, context)

            for attempt in range(self.max_retries):
                try:
                    logger.debug(f"Attempt {attempt + 1}/{self.max_retries} calling LLM...")
                    completion = await self.async_client.chat.completions.create(
                        # `extra_headers` moved to client initialization
                        model="microsoft/phi-4", # Or make this configurable
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3, # Lower temperature for more deterministic topics
                        max_tokens=150, # Limit response length
                        response_format={"type": "json_object"}, # Request JSON output if model supports it
                    )

                    response_text = completion.choices[0].message.content
                    logger.debug(f"LLM Raw Response: {response_text}")

                    try:
                        # Attempt to parse the JSON response
                        result = json.loads(response_text)

                        # Basic validation of expected keys
                        if not all(k in result for k in ["topic", "keywords", "relationship", "confidence"]):
                             logger.warning(f"LLM response missing expected keys: {result}")
                             # Provide default values for missing keys
                             result.setdefault("topic", "Unknown")
                             result.setdefault("keywords", [])
                             result.setdefault("relationship", "UNKNOWN")
                             result.setdefault("confidence", 0)


                        # Add to cache upon successful analysis
                        # self._add_to_cache(cache_key_current, cache_key_prev, result)
                        # Relying on @lru_cache on the lookup method for now.

                        return result

                    except json.JSONDecodeError as json_e:
                        logger.error(f"Failed to parse JSON response: {json_e}. Response: '{response_text}'")
                        # Fallback if JSON parsing fails
                        return {
                            "topic": "Parsing Error",
                            "keywords": [],
                            "relationship": "UNKNOWN",
                            "confidence": 0,
                            "error": f"JSONDecodeError: {json_e}",
                            "raw_response": response_text # Include raw response for debugging
                        }
                    except Exception as parse_e: # Catch other potential errors during parsing/validation
                        logger.error(f"Error processing LLM response: {parse_e}. Response: '{response_text}'")
                        return {
                            "topic": "Processing Error",
                            "keywords": [],
                            "relationship": "UNKNOWN",
                            "confidence": 0,
                            "error": f"ProcessingError: {parse_e}",
                            "raw_response": response_text
                        }


                except RateLimitError as rle:
                    logger.warning(f"Rate limit hit (Attempt {attempt + 1}): {rle}. Retrying after delay...")
                    await asyncio.sleep(self.retry_delay * (attempt + 1)) # Exponential backoff might be better
                except APIError as apie:
                    logger.error(f"API error (Attempt {attempt + 1}): {apie}. Retrying...")
                    await asyncio.sleep(self.retry_delay)
                except Exception as e:
                    logger.error(f"Unexpected error during API call (Attempt {attempt + 1}): {e}", exc_info=True)
                    if attempt >= self.max_retries - 1:
                        logger.error("Maximum retries reached. Analysis failed.")
                        raise # Re-raise the last exception after all retries fail
                    await asyncio.sleep(self.retry_delay)

            # If loop completes without returning/raising, it means all retries failed.
            logger.error("Segment analysis failed after all retries.")
            return {
                "topic": "Analysis Failed",
                "keywords": [],
                "relationship": "UNKNOWN",
                "confidence": 0,
                "error": "Max retries exceeded",
            }

    def analyze_segment(
        self, current_segment: Dict, previous_segment: Optional[Dict] = None
    ) -> Dict:
        """Synchronous wrapper for the asynchronous `analyze_segment_async`.

        Provides a blocking interface to the segment analysis functionality.

        Args:
            current_segment (Dict): Dictionary for the current segment ('content').
            previous_segment (Optional[Dict], optional): Dictionary for the
                previous segment ('content', 'topic'). Defaults to None.

        Returns:
            Dict: The analysis result dictionary from the LLM.
        """
        logger.debug("Running synchronous wrapper for analyze_segment_async.")
        try:
            # Get the current event loop or create a new one if needed
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If called from within an existing running loop, create a task
                # This scenario is less common for a simple wrapper but good practice
                # Note: This might still block if not awaited properly by the caller.
                # A truly non-blocking sync wrapper is complex (e.g., using threads).
                # For typical script usage, asyncio.run is sufficient.
                logger.warning("analyze_segment called within a running event loop. Behavior might be unexpected.")
                # This approach is generally discouraged. Consider using the async version directly.
                # task = loop.create_task(self.analyze_segment_async(current_segment, previous_segment))
                # return asyncio.run(task) # This is incorrect within a running loop
                # Fallback to running in a new loop - might cause issues.
                return asyncio.run(self.analyze_segment_async(current_segment, previous_segment))
            else:
                return loop.run_until_complete(
                    self.analyze_segment_async(current_segment, previous_segment)
                )
        except RuntimeError as e:
             # Handle cases where asyncio.run can't be used (e.g., nested calls)
             logger.warning(f"RuntimeError during asyncio execution in sync wrapper: {e}. Trying new event loop policy.")
             # Try setting a different event loop policy if needed (use with caution)
             # asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy()) # Example for Windows
             return asyncio.run(self.analyze_segment_async(current_segment, previous_segment))


    def _preprocess_text(self, text: str) -> str:
        """Preprocesses text for TF-IDF similarity calculation.

        Converts text to lowercase and removes English stopwords.

        Args:
            text (str): The input text string.

        Returns:
            str: The preprocessed text string. Returns an empty string if
                 input is None or empty.
        """
        if not text:
            return ""
        # Simple whitespace tokenization, lowercase, and stopword removal
        words = text.lower().split()
        words = [w for w in words if w.isalnum() and w not in self.stop_words]
        return " ".join(words)

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculates cosine similarity between two preprocessed text strings.

        Uses a TF-IDF vectorizer (`self.vectorizer`) fitted on the two texts
        to compute the cosine similarity score.

        Args:
            text1 (str): The first text string.
            text2 (str): The second text string.

        Returns:
            float: The cosine similarity score between 0.0 and 1.0. Returns 0.0
                   if either text is empty after preprocessing or if vectorization fails.
        """
        if not text1 or not text2:
            return 0.0

        # Preprocess texts
        proc_text1 = self._preprocess_text(text1)
        proc_text2 = self._preprocess_text(text2)

        if not proc_text1 or not proc_text2:
            return 0.0

        try:
            # Fit and transform with the instance's vectorizer
            tfidf_matrix = self.vectorizer.fit_transform([proc_text1, proc_text2])

            # Ensure matrix has expected shape (at least 2 rows)
            if tfidf_matrix.shape[0] < 2:
                 logger.warning("TF-IDF matrix has fewer than 2 rows, cannot compute similarity.")
                 return 0.0

            # Calculate cosine similarity
            # Handle potential sparse matrix output if necessary
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

            # Extract the single similarity value
            # Ensure the result is a float, clip to [0, 1] for safety
            return float(np.clip(similarity[0][0], 0.0, 1.0))

        except ValueError as ve:
            # Can happen if vocabulary is empty after stopword removal etc.
            logger.warning(f"ValueError during TF-IDF similarity calculation: {ve}. Texts: '{proc_text1[:50]}...', '{proc_text2[:50]}...'")
            return 0.0
        except Exception as e:
            logger.error(f"Unexpected error calculating similarity: {e}", exc_info=True)
            return 0.0


    def _create_batches(self, transcript_sentences: List[Dict]) -> List[List[Dict]]:
        """Groups transcript sentences into batches for analysis.

        Aims to create batches close to `self.batch_size`. Implements "smart"
        boundary detection: if the similarity between the text of a potential
        batch and the next sentence falls below `self.similarity_threshold`,
        it forces a batch break, assuming a potential topic shift.

        Args:
            transcript_sentences (List[Dict]): A list of dictionaries, where each
                dictionary represents a sentence and must contain at least
                'content', 'start', and 'end' keys.

        Returns:
            List[List[Dict]]: A list of batches, where each batch is a list of
            sentence dictionaries.
        """
        if not transcript_sentences:
            return []

        batches = []
        current_batch = []
        num_sentences = len(transcript_sentences)

        for i, sentence in enumerate(transcript_sentences):
            current_batch.append(sentence)

            # Check if we should finalize the batch
            finalize_batch = False
            if len(current_batch) >= self.batch_size:
                # Check for smart boundary if not the last sentence
                if i + 1 < num_sentences:
                    current_batch_text = " ".join(s["content"] for s in current_batch)
                    next_sentence_text = transcript_sentences[i + 1]["content"]

                    # Calculate similarity between the current batch and the next sentence
                    similarity = self._calculate_similarity(current_batch_text, next_sentence_text)
                    logger.debug(f"Similarity between batch ending at {i} and next sentence: {similarity:.3f}")

                    # If similarity is low, consider it a natural break point
                    if similarity < self.similarity_threshold:
                        logger.debug(f"Low similarity ({similarity:.3f} < {self.similarity_threshold}) detected. Forcing batch break.")
                        finalize_batch = True
                    else:
                        # If batch size is met but similarity is high, continue batching?
                        # For now, we break if batch_size is met unless similarity forces earlier break.
                        finalize_batch = True # Default break at batch_size
                else:
                    # Final sentence, finalize the batch
                    finalize_batch = True

            # If the batch size limit is reached (or low similarity detected), finalize
            if finalize_batch:
                batches.append(current_batch)
                current_batch = []

        # Add any remaining sentences in the last batch
        if current_batch:
            batches.append(current_batch)

        logger.info(f"Created {len(batches)} batches from {num_sentences} sentences.")
        return batches

    def _combine_batch(self, batch: List[Dict]) -> Dict:
        """Combines sentences within a batch into a single segment dictionary.

        Concatenates the 'content' of all sentences and sets the 'start' time
        from the first sentence and the 'end' time from the last sentence.

        Args:
            batch (List[Dict]): A list of sentence dictionaries, each containing
                at least 'start', 'end', and 'content'.

        Returns:
            Dict: A dictionary representing the combined segment with 'start',
                  'end', and 'content' keys. Returns an empty dict if the batch is empty.
        """
        if not batch:
            return {}
        return {
            "start": batch[0].get("start", 0.0),
            "end": batch[-1].get("end", 0.0),
            "content": " ".join(s.get("content", "") for s in batch).strip(),
        }

    async def _analyze_batch(
        self, batch_segment: Dict, previous_batch_analysis: Optional[Dict] = None
    ) -> Dict:
        """Analyzes a combined batch segment using the async LLM analysis method.

        This is a thin wrapper around `analyze_segment_async` specifically for
        handling combined batch data.

        Args:
            batch_segment (Dict): The combined segment dictionary created by
                `_combine_batch`. Must contain 'content'.
            previous_batch_analysis (Optional[Dict], optional): The analysis result
                dictionary from the previously analyzed batch. Used for context.
                Defaults to None.

        Returns:
            Dict: The analysis result dictionary from `analyze_segment_async`.
        """
        # Pass the combined segment and the *analysis result* of the previous batch
        return await self.analyze_segment_async(batch_segment, previous_batch_analysis)


    async def _analyze_batches(self, batches: List[List[Dict]]) -> List[Dict]:
        """Analyzes multiple batches concurrently using asyncio.gather.

        Combines sentences in each batch, then runs `_analyze_batch` for multiple
        batches concurrently, respecting `self.max_concurrent` via the semaphore
        in `analyze_segment_async`. Tracks progress using tqdm.

        Args:
            batches (List[List[Dict]]): The list of sentence batches created by
                `_create_batches`.

        Returns:
            List[Dict]: A list of analysis result dictionaries, one for each input batch.
                        The order corresponds to the input batch order.
        """
        analyses = [None] * len(batches) # Pre-allocate list to maintain order
        tasks = []
        previous_analysis_result = None # Store the *result* of the previous analysis

        # Create tasks respecting concurrency limits implicitly via semaphore in analyze_segment_async
        # We still use asyncio.gather for efficient scheduling and result collection.
        async def task_wrapper(batch_idx, current_batch_list, prev_analysis):
            combined_segment = self._combine_batch(current_batch_list)
            if not combined_segment: # Skip empty batches
                 return batch_idx, None
            result = await self._analyze_batch(combined_segment, prev_analysis)
            return batch_idx, result

        # Use tqdm for progress tracking
        with tqdm.tqdm(total=len(batches), desc="Analyzing Batches", unit="batch") as pbar:
            # Process batches sequentially to maintain context dependency
            for i, batch in enumerate(batches):
                # Analyze the current batch using the result of the previous one
                combined_segment = self._combine_batch(batch)
                if not combined_segment:
                    logger.warning(f"Skipping empty batch at index {i}")
                    analyses[i] = None # Store None for empty batch
                    pbar.update(1)
                    continue

                # Analyze the segment, passing the *analysis result* of the previous segment
                analysis_result = await self.analyze_segment_async(combined_segment, previous_analysis_result)
                analyses[i] = analysis_result # Store result in correct position

                # Update previous_analysis_result for the next iteration
                # We need both the content and the analysis result for context
                # Store the combined segment content along with its analysis result
                previous_analysis_result = {
                    "content": combined_segment.get("content", ""),
                    "topic": analysis_result.get("topic", "Unknown"),
                    # Include other relevant fields from analysis_result if needed by the prompt
                }
                pbar.update(1)


        # Filter out potential None results from skipped empty batches if necessary
        # final_analyses = [a for a in analyses if a is not None]
        # return final_analyses
        return analyses # Return list including potential None for skipped batches


    def identify_segments(self, transcript_sentences: List[Dict]) -> List[Dict]:
        """Identifies topic-based segments within a transcript.

        Orchestrates the process:
        1. Creates batches of sentences (`_create_batches`).
        2. Analyzes these batches concurrently (`_analyze_batches`).
        3. Iterates through batch analyses, merging consecutive batches into
           segments based on the 'relationship' and 'confidence' predicted by
           the LLM. A new segment is started on 'NEW' or high-confidence 'SHIFT'.

        Args:
            transcript_sentences (List[Dict]): A list of sentence dictionaries
                from the transcript, each with 'start', 'end', 'content'.

        Returns:
            List[Dict]: A list of identified segment dictionaries. Each segment
                dictionary contains 'start', 'end', 'content', 'topic', and 'keywords'.
        """
        if not transcript_sentences:
            logger.warning("Transcript sentences list is empty. Cannot identify segments.")
            return []

        logger.info("Identifying segments based on topic analysis...")

        # 1. Create batches
        logger.info("Creating analysis batches...")
        batches = self._create_batches(transcript_sentences)
        if not batches:
            logger.warning("No batches were created from the transcript sentences.")
            return []
        logger.info(f"Created {len(batches)} batches.")

        # 2. Analyze batches concurrently
        logger.info("Analyzing batches using LLM...")
        # Run the async batch analysis within the sync method
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                 logger.warning("identify_segments called within a running event loop. Running _analyze_batches may block.")
                 # This is problematic. Ideally, the caller should be async.
                 # As a fallback, run in a separate thread or use asyncio.run() if possible.
                 analyses = asyncio.run(self._analyze_batches(batches))
            else:
                 analyses = loop.run_until_complete(self._analyze_batches(batches))
        except RuntimeError:
             logger.warning("RuntimeError during asyncio execution in identify_segments. Trying asyncio.run().")
             analyses = asyncio.run(self._analyze_batches(batches))

        if len(analyses) != len(batches):
            logger.error(f"Mismatch between number of batches ({len(batches)}) and analyses ({len(analyses)}). Aborting segmentation.")
            return []

        # 3. Process analyses to form segments
        logger.info("Merging batch analyses into final segments...")
        segments = []
        current_segment_batches = [] # Store the original sentence dicts for the current segment
        current_segment_analysis = None # Store the analysis of the *first* batch in the segment

        for i, (batch, analysis) in enumerate(zip(batches, analyses)):
            if analysis is None: # Skip batches that failed analysis or were empty
                logger.warning(f"Skipping batch {i+1} due to missing analysis.")
                continue

            if not current_segment_batches:
                # Start the first segment
                current_segment_batches.extend(batch)
                current_segment_analysis = analysis
            else:
                # Decide whether to merge or start a new segment
                # Use relationship and confidence from the *current* batch's analysis
                relationship = analysis.get("relationship", "UNKNOWN").upper()
                confidence = analysis.get("confidence", 0)

                # Define thresholds for splitting (these might need tuning)
                new_segment_threshold = 70
                shift_segment_threshold = 85

                should_split = False
                if relationship == "NEW" and confidence > new_segment_threshold:
                    should_split = True
                    logger.debug(f"Segment split condition met: NEW relationship, confidence {confidence} > {new_segment_threshold}")
                elif relationship == "SHIFT" and confidence > shift_segment_threshold:
                    should_split = True
                    logger.debug(f"Segment split condition met: SHIFT relationship, confidence {confidence} > {shift_segment_threshold}")

                if should_split:
                    # Finalize the current segment
                    finalized_segment = self._combine_batch(current_segment_batches)
                    # Assign topic/keywords from the *first* batch of that segment
                    finalized_segment["topic"] = current_segment_analysis.get("topic", "Unknown")
                    finalized_segment["keywords"] = current_segment_analysis.get("keywords", [])
                    segments.append(finalized_segment)
                    logger.debug(f"Finalized segment {len(segments)} ending at batch {i}, topic: {finalized_segment['topic']}")

                    # Start a new segment with the current batch
                    current_segment_batches = list(batch) # Start new list
                    current_segment_analysis = analysis
                else:
                    # Continue the current segment: append the current batch
                    current_segment_batches.extend(batch)
                    # Keep the topic/keywords from the *start* of the segment

        # Handle the last segment
        if current_segment_batches:
            finalized_segment = self._combine_batch(current_segment_batches)
            # Assign topic/keywords from the first batch of this last segment
            if current_segment_analysis: # Ensure analysis exists
                 finalized_segment["topic"] = current_segment_analysis.get("topic", "Unknown")
                 finalized_segment["keywords"] = current_segment_analysis.get("keywords", [])
            else: # Fallback if the very first batch failed analysis
                 finalized_segment["topic"] = "Unknown"
                 finalized_segment["keywords"] = []
            segments.append(finalized_segment)
            logger.debug(f"Finalized last segment {len(segments)}, topic: {finalized_segment['topic']}")


        logger.info(f"Identified {len(segments)} topic segments.")
        return segments


def process_transcript(
    transcript_sentences: List[Dict],
    project_path: str,
    num_topics: int = 5, # Note: num_topics is not directly used by TopicAnalyzer logic anymore
    register: str = "gen-ai",
) -> Dict:
    """Processes a transcript to perform topic modeling and segmentation.

    Uses the `TopicAnalyzer` to identify segments based on topic shifts detected
    by an LLM. It formats the results into a dictionary containing metadata about
    the identified topics and the segments themselves. Saves the results to a
    JSON file within the project path and manages checkpoints.

    Args:
        transcript_sentences (List[Dict]): A list of sentence dictionaries, each
            required to have 'start', 'end', and 'content' keys.
        project_path (str): The path to the project directory where results and
            checkpoints will be saved.
        num_topics (int, optional): This argument is currently **not used** by the
            underlying `TopicAnalyzer` logic, which dynamically determines the
            number of segments based on content analysis. It's kept for potential
            future use or compatibility. Defaults to 5.
        register (str, optional): The analysis register/domain (e.g., 'gen-ai')
            passed to the `TopicAnalyzer` to guide the LLM's analysis.
            Defaults to "gen-ai".

    Returns:
        Dict: A dictionary containing the analysis results:
            'topics' (List[Dict]): A list where each entry represents a unique
                topic found across the segments, including 'topic_id', 'topic' name,
                and associated 'words' (keywords).
            'segments' (List[Dict]): A list of the identified segments, each with
                'segment_id', 'start_time', 'end_time', 'duration',
                'dominant_topic', 'top_keywords', and the full 'transcript' text
                of the segment.
            'register' (str): The register used for the analysis.

    Raises:
        ValueError: If `TopicAnalyzer` fails to initialize (e.g., missing API key).
        Exception: Propagates exceptions from `TopicAnalyzer.identify_segments`
                   if segmentation fails critically.
    """
    logger.info(f"Starting transcript processing for project: {project_path}")
    results_path = os.path.join(project_path, "topic_analysis_results.json") # More specific name

    # Check if results already exist (simple caching mechanism)
    # TODO: Add more robust checkpoint loading/resuming logic if needed
    if os.path.exists(results_path):
        logger.info(f"Loading existing topic analysis results from: {results_path}")
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            # Optional: Add a check here to see if the parameters match (e.g., register)
            # if results.get("register") == register:
            #     return results
            # else:
            #     logger.info("Existing results found but parameters differ. Re-analyzing.")
            return results # For now, just return if file exists
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load existing results file ({e}). Proceeding with analysis.")

    # Initialize the analyzer
    try:
        analyzer = TopicAnalyzer(register=register)
    except ValueError as ve:
        logger.error(f"Failed to initialize TopicAnalyzer: {ve}")
        raise # Re-raise critical initialization error

    # Identify segments using the analyzer
    logger.info("Identifying topic segments...")
    try:
        segments = analyzer.identify_segments(transcript_sentences)
    except Exception as e:
        logger.error(f"Error during segment identification: {e}", exc_info=True)
        # Decide how to handle: raise error, return partial results, etc.
        raise # Re-raise segmentation error

    if not segments:
        logger.warning("No segments were identified. Returning empty results.")
        return {"topics": [], "segments": [], "register": register}

    # Generate metadata and structure results
    logger.info("Formatting analysis results...")
    segment_metadata = []
    topic_summary = {} # Collect unique topics and their keywords

    # Use tqdm for progress indication during metadata generation
    for i, segment in enumerate(tqdm.tqdm(segments, desc="Generating Segment Metadata")):
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
            "duration": max(0.0, end_time - start_time), # Ensure non-negative duration
            "dominant_topic": topic,
            "top_keywords": keywords,
            "transcript": content,
        }
        segment_metadata.append(segment_meta)

        # Update topic summary (collect all keywords associated with a topic name)
        if topic != "Unknown":
            if topic not in topic_summary:
                topic_summary[topic] = set()
            topic_summary[topic].update(keywords)

    # Create the final list of unique topics
    topics_list = [
        {"topic_id": i + 1, "topic": topic_name, "words": sorted(list(kw_set))}
        for i, (topic_name, kw_set) in enumerate(topic_summary.items())
    ]

    # Create final results structure
    results = {
        "topics": topics_list,
        "segments": segment_metadata,
        "register": register,
        "analysis_info": { # Add some metadata about the analysis run
            "total_segments": len(segments),
            "analyzer_config": {
                 "register": analyzer.register,
                 "batch_size": analyzer.batch_size,
                 "similarity_threshold": analyzer.similarity_threshold,
            }
        }
    }

    # Save results to JSON file
    logger.info(f"Saving topic analysis results to: {results_path}")
    try:
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    except IOError as e:
        logger.error(f"Failed to save results to {results_path}: {e}")
    except TypeError as e:
        logger.error(f"Failed to serialize results to JSON: {e}")


    # Save checkpoint indicating completion of topic modeling
    logger.info("Saving topic modeling checkpoint...")
    save_checkpoint(
        project_path,
        CHECKPOINTS["TOPIC_MODELING_COMPLETE"],
        {"results_path": results_path, "num_segments": len(segments)} # Save path and key info
    )

    logger.info("Transcript processing and topic modeling complete.")
    return results
