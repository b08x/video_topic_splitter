# analysis/segment_analysis.py
#!/usr/bin/env python3
"""
Contains functions for analyzing segmented video clips using the Gemini API.
"""

import logging
from typing import Dict, Any
import json  # Import json for parsing

# Assuming GeminiClient is correctly imported from its location
# Adjust the relative import path if necessary
try:
    from ..api.gemini import GeminiClient
except ImportError:
    # Fallback or absolute path if relative import fails
    from video_topic_splitter.api.gemini import GeminiClient

logger = logging.getLogger(__name__)


class SegmentAnalyzer:
    """
    Analyzes video segments using the Gemini API.
    """

    def __init__(self, gemini_client: GeminiClient):
        """
        Initializes the SegmentAnalyzer with a GeminiClient instance.

        Args:
            gemini_client: An instance of the GeminiClient for API interactions.
        """
        self.gemini_client = gemini_client

    def analyze_segment(self, segment_path: str) -> Dict[str, Any]:
        """
        Analyzes a video segment using the Gemini API.

        Args:
            segment_path: Path to the video segment file.

        Returns:
            A dictionary containing the analysis results.
        """
        logger.info(f"Analyzing segment: {segment_path}")
        try:
            # Construct the prompt for Gemini
            # Consider if including the segment_path in the prompt is necessary
            # or if it's only needed for the API call itself.
            prompt = (
                "Summarize this video segment and identify the main topics discussed."
            )
            # Assuming the analyze method handles the file path separately
            analysis_text = self.gemini_client.analyze(prompt, segment_path)

            # Process the analysis text (e.g., extract key information, format it)
            analysis_results = self._process_analysis_text(analysis_text)
            return analysis_results

        except Exception as e:
            logger.error(
                f"Error analyzing segment {segment_path}: {e}", exc_info=True)
            # Return a dictionary with an error key for consistency
            return {"error": f"Failed to analyze segment: {e}"}

    def _process_analysis_text(self, analysis_text: str) -> Dict[str, Any]:
        """
        Processes the raw text analysis from the Gemini API.

        Args:
            analysis_text: The raw text response from the Gemini API.

        Returns:
            A dictionary containing the processed analysis results.
        """
        # Attempt to parse as JSON first, assuming a structured response
        try:
            data = json.loads(analysis_text)
            # Basic validation: Check if it's a dict and has expected keys
            if isinstance(data, dict) and "summary" in data and "topics" in data:
                # Ensure topics is a list
                if not isinstance(data["topics"], list):
                    logger.warning(
                        "Gemini analysis 'topics' field is not a list. Wrapping.")
                    # Convert/wrap non-list topics
                    data["topics"] = [str(data["topics"])]
                return data
            else:
                logger.warning(
                    "Parsed JSON from Gemini analysis lacks expected structure. Treating as plain text.")
                # Fall through to plain text handling
        except json.JSONDecodeError:
            # If JSON parsing fails, treat it as plain text
            logger.debug(
                "Gemini analysis response is not valid JSON. Treating as plain text summary.")
            # Basic structure for plain text response
            analysis = {"summary": analysis_text.strip(), "topics": []}
            # Optionally, try simple keyword extraction from the summary as topics
            # (e.g., using regex or simple splitting - could be unreliable)
            return analysis
        except Exception as e:
            logger.error(
                f"Error processing Gemini analysis text: {e}", exc_info=True)
            return {"summary": "Analysis processing failed", "topics": [], "error": str(e)}

        # Fallback if JSON parsing succeeded but structure was wrong
        analysis = {"summary": analysis_text.strip(), "topics": []}
        return analysis
