"""Core topic analysis functionality."""

import asyncio
import logging
import os
from typing import Dict, List, Optional

import tqdm
from openai import AsyncOpenAI, OpenAI, RateLimitError, APIError

from ..prompt_templates import get_topic_prompt
from .topic_analyzer_config import TopicAnalyzerConfig
from .async_cache import AsyncCache
from .segment_batcher import SegmentBatcher
from .response_parser import ResponseParser

logger = logging.getLogger(__name__)


class TopicAnalyzer:
    """Core topic analyzer with modular architecture."""

    def __init__(self, config: Optional[TopicAnalyzerConfig] = None):
        """Initialize the TopicAnalyzer with configuration.
        
        Args:
            config: Configuration object. If None, uses default configuration.
            
        Raises:
            ValueError: If OPENROUTER_API_KEY environment variable is not set.
        """
        self.config = config or TopicAnalyzerConfig()
        
        # Validate API key
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")
        
        # Configure OpenRouter clients
        default_headers = {
            "HTTP-Referer": "https://github.com/your-repo/video-topic-splitter",
            "X-Title": "Video Topic Splitter",
        }
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers=default_headers,
            max_retries=0,
        )
        self.async_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers=default_headers,
            max_retries=0,
        )
        
        # Initialize components
        self.cache = AsyncCache(self.config.cache_size)
        self.batcher = SegmentBatcher(self.config)
        self.parser = ResponseParser()
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        if self.config.debug:
            logging.getLogger(__name__).setLevel(logging.DEBUG)
        
        logger.info(
            f"TopicAnalyzer initialized: register='{self.config.register}', "
            f"batch_size={self.config.batch_size}, max_concurrent={self.config.max_concurrent}"
        )

    def _create_cache_key(self, current_content: str, prev_content: Optional[str] = None) -> str:
        """Create a cache key for the given content."""
        # Use hash for long content to avoid memory issues
        current_key = current_content if len(current_content) < 100 else str(hash(current_content))
        prev_key = prev_content if prev_content and len(prev_content) < 100 else str(hash(prev_content)) if prev_content else ""
        return f"{current_key}:{prev_key}"

    async def analyze_segment_async(
        self, current_segment: Dict, previous_segment: Optional[Dict] = None
    ) -> Dict:
        """Analyze a text segment asynchronously using the LLM with caching."""
        current_content = current_segment.get("content", "")
        prev_content = previous_segment.get("content", "") if previous_segment else None
        prev_topic = previous_segment.get("topic", "Unknown") if previous_segment else "None"

        # Check cache first
        cache_key = self._create_cache_key(current_content, prev_content)
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            logger.debug("Cache hit for segment analysis.")
            return cached_result

        logger.debug("Cache miss. Analyzing segment via API.")
        
        # Acquire semaphore for concurrency control
        async with self.semaphore:
            # Build context
            context = self._build_context(current_content, prev_content, prev_topic)
            prompt = get_topic_prompt(self.config.register, context)

            # Call API with retries
            for attempt in range(self.config.max_retries):
                try:
                    logger.debug(f"Attempt {attempt + 1}/{self.config.max_retries} calling LLM...")
                    
                    completion = await self.async_client.chat.completions.create(
                        model=self.config.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        response_format={"type": "json_object"},
                    )

                    response_text = completion.choices[0].message.content
                    if self.config.debug:
                        logger.debug(f"LLM Raw Response: {response_text}")

                    # Parse and validate response
                    parsed_result = self.parser.parse_json_response(response_text)
                    if parsed_result:
                        validated_result = self.parser.validate_response(parsed_result)
                        
                        # Cache the result
                        await self.cache.set(cache_key, validated_result)
                        
                        return validated_result
                    else:
                        logger.warning(f"Could not parse JSON from response (attempt {attempt + 1})")
                        if attempt < self.config.max_retries - 1:
                            await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                            continue
                        
                        # Return default on final failure
                        return self.parser.validate_response({
                            "topic": "Parsing Error",
                            "keywords": [],
                            "relationship": "NEW",
                            "confidence": 0
                        })

                except RateLimitError as rle:
                    logger.warning(f"Rate limit hit (Attempt {attempt + 1}): {rle}")
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                except APIError as apie:
                    logger.error(f"API error (Attempt {attempt + 1}): {apie}")
                    await asyncio.sleep(self.config.retry_delay)
                except Exception as e:
                    logger.error(f"Unexpected error during API call (Attempt {attempt + 1}): {e}", exc_info=True)
                    if attempt >= self.config.max_retries - 1:
                        logger.error("Maximum retries reached. Analysis failed.")
                        raise
                    await asyncio.sleep(self.config.retry_delay)

            # All retries failed
            logger.error("Segment analysis failed after all retries.")
            return {
                "topic": "Analysis Failed",
                "keywords": [],
                "relationship": "UNKNOWN",
                "confidence": 0,
                "error": "Max retries exceeded",
            }
    
    def _build_context(self, current_content: str, prev_content: Optional[str], prev_topic: str) -> str:
        """Build context string for the LLM prompt."""
        context = ""
        
        if prev_content:
            # Truncate previous content if too long
            max_prev_len = self.config.max_prev_context_length
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

        # Truncate current content if too long
        max_curr_len = self.config.max_current_content_length
        truncated_current_content = (
            (current_content[:max_curr_len] + "...")
            if len(current_content) > max_curr_len
            else current_content
        )
        context += f"Analyze the following current segment:\nContent: {truncated_current_content}"
        
        return context

    def analyze_segment(
        self, current_segment: Dict, previous_segment: Optional[Dict] = None
    ) -> Dict:
        """Synchronous wrapper for the asynchronous analyze_segment_async."""
        logger.debug("Running synchronous wrapper for analyze_segment_async.")
        try:
            return asyncio.run(self.analyze_segment_async(current_segment, previous_segment))
        except RuntimeError:
            # Handle cases where asyncio.run can't be used (e.g., nested calls)
            logger.warning("RuntimeError during asyncio execution in sync wrapper.")
            try:
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(
                    self.analyze_segment_async(current_segment, previous_segment)
                )
            except RuntimeError:
                # Last resort - create new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(
                        self.analyze_segment_async(current_segment, previous_segment)
                    )
                finally:
                    loop.close()

    async def _analyze_batches(self, batches: List[List[Dict]]) -> List[Dict]:
        """Analyze multiple batches sequentially with progress tracking."""
        analyses = []
        previous_analysis_result = None
        
        with tqdm.tqdm(total=len(batches), desc="Analyzing Batches", unit="batch") as pbar:
            for i, batch in enumerate(batches):
                # Combine batch into segment
                combined_segment = self.batcher.combine_batch(batch)
                if not combined_segment:
                    logger.warning(f"Skipping empty batch at index {i}")
                    analyses.append(None)
                    pbar.update(1)
                    continue
                
                # Analyze segment
                analysis_result = await self.analyze_segment_async(combined_segment, previous_analysis_result)
                analyses.append(analysis_result)
                
                # Update context for next analysis
                previous_analysis_result = {
                    "content": combined_segment.get("content", ""),
                    "topic": analysis_result.get("topic", "Unknown"),
                }
                pbar.update(1)
        
        return analyses

    def identify_segments(self, transcript_sentences: List[Dict]) -> List[Dict]:
        """Identify topic-based segments within a transcript."""
        if not transcript_sentences:
            logger.warning("Transcript sentences list is empty. Cannot identify segments.")
            return []

        logger.info("Identifying segments based on topic analysis...")

        # Create batches using the batcher
        logger.info("Creating analysis batches...")
        batches = self.batcher.create_batches(transcript_sentences)
        if not batches:
            logger.warning("No batches were created from the transcript sentences.")
            return []

        logger.info(f"Created {len(batches)} batches.")

        # Analyze batches
        logger.info("Analyzing batches using LLM...")
        try:
            analyses = asyncio.run(self._analyze_batches(batches))
        except RuntimeError:
            logger.warning("RuntimeError during batch analysis. Trying alternative approach.")
            try:
                loop = asyncio.get_event_loop()
                analyses = loop.run_until_complete(self._analyze_batches(batches))
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    analyses = loop.run_until_complete(self._analyze_batches(batches))
                finally:
                    loop.close()

        if len(analyses) != len(batches):
            logger.error(f"Mismatch between batches ({len(batches)}) and analyses ({len(analyses)})")
            return []

        # Process analyses to form segments
        logger.info("Merging batch analyses into final segments...")
        segments = self._merge_analyses_into_segments(batches, analyses)
        
        logger.info(f"Identified {len(segments)} topic segments.")
        return segments
    
    def _merge_analyses_into_segments(self, batches: List[List[Dict]], analyses: List[Dict]) -> List[Dict]:
        """Merge batch analyses into final segments based on topic relationships."""
        segments = []
        current_segment_batches = []
        current_segment_analysis = None
        
        # Thresholds for segment splitting
        new_segment_threshold = 70
        shift_segment_threshold = 85
        
        for i, (batch, analysis) in enumerate(zip(batches, analyses)):
            if analysis is None:
                logger.warning(f"Skipping batch {i+1} due to missing analysis.")
                continue
            
            if not current_segment_batches:
                # Start the first segment
                current_segment_batches.extend(batch)
                current_segment_analysis = analysis
            else:
                # Check if we should split
                relationship = analysis.get("relationship", "UNKNOWN").upper()
                confidence = analysis.get("confidence", 0)
                
                should_split = False
                if relationship == "NEW" and confidence > new_segment_threshold:
                    should_split = True
                    logger.debug(f"Segment split: NEW relationship, confidence {confidence}")
                elif relationship == "SHIFT" and confidence > shift_segment_threshold:
                    should_split = True
                    logger.debug(f"Segment split: SHIFT relationship, confidence {confidence}")
                
                if should_split:
                    # Finalize current segment
                    finalized_segment = self.batcher.combine_batch(current_segment_batches)
                    finalized_segment["topic"] = current_segment_analysis.get("topic", "Unknown")
                    finalized_segment["keywords"] = current_segment_analysis.get("keywords", [])
                    segments.append(finalized_segment)
                    
                    # Start new segment
                    current_segment_batches = list(batch)
                    current_segment_analysis = analysis
                else:
                    # Continue current segment
                    current_segment_batches.extend(batch)
        
        # Handle last segment
        if current_segment_batches and current_segment_analysis:
            finalized_segment = self.batcher.combine_batch(current_segment_batches)
            finalized_segment["topic"] = current_segment_analysis.get("topic", "Unknown")
            finalized_segment["keywords"] = current_segment_analysis.get("keywords", [])
            segments.append(finalized_segment)
        
        return segments