"""Topic modeling using OpenRouter with microsoft/phi-4 model."""

import os
import json
import time
from typing import Dict, List, Optional
import progressbar
from openai import OpenAI

from .constants import CHECKPOINTS
from .project import save_checkpoint


class TopicAnalyzer:
    """Class to handle topic analysis using OpenRouter's phi-4 model."""
    
    def __init__(self, max_retries: int = 3, retry_delay: int = 5):
        """Initialize the TopicAnalyzer."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")
            
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def analyze_segment(self, current_segment: Dict, previous_segment: Optional[Dict] = None) -> Dict:
        """Analyze a segment considering previous context."""
        context = ""
        if previous_segment:
            context = (f"Previous segment context:\n"
                      f"Content: {previous_segment['content']}\n"
                      f"Topic: {previous_segment.get('topic', 'Unknown')}\n\n")

        prompt = f"""
        {context}
        Analyze the following segment:
        {current_segment['content']}

        Provide a structured analysis with:
        1. Main topic (single phrase)
        2. 5-10 relevant keywords
        3. Topic relationship to previous segment (if provided):
           - CONTINUATION: same topic
           - SHIFT: related but different
           - NEW: completely new topic
        4. Confidence score (0-100)

        Format response as JSON:
        {{
            "topic": "main topic",
            "keywords": ["keyword1", "keyword2", ...],
            "relationship": "CONTINUATION|SHIFT|NEW",
            "confidence": 85
        }}
        """

        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
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
                    return json.loads(json_str)
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
                    time.sleep(self.retry_delay)
                else:
                    print(f"All attempts failed: {str(e)}")
                    raise

    def identify_segments(self, transcript: List[Dict]) -> List[Dict]:
        """Identify segments based on topic analysis."""
        print("Identifying segments based on topics...")
        segments = []
        current_segment = None
        previous_segment = None

        for sentence in progressbar.progressbar(transcript):
            if not current_segment:
                current_segment = {
                    "start": sentence["start"],
                    "end": sentence["end"],
                    "content": sentence["content"],
                }
            else:
                # Analyze current content with context
                analysis = self.analyze_segment(
                    {"content": current_segment["content"]},
                    previous_segment
                )
                
                # Decide whether to start a new segment
                if (analysis["relationship"] == "NEW" and analysis["confidence"] > 70) or \
                   (analysis["relationship"] == "SHIFT" and analysis["confidence"] > 85):
                    # Finalize current segment
                    current_segment["end"] = sentence["start"]
                    current_segment["topic"] = analysis["topic"]
                    current_segment["keywords"] = analysis["keywords"]
                    segments.append(current_segment)
                    
                    # Start new segment
                    previous_segment = current_segment
                    current_segment = {
                        "start": sentence["start"],
                        "end": sentence["end"],
                        "content": sentence["content"],
                    }
                else:
                    # Continue current segment
                    current_segment["end"] = sentence["end"]
                    current_segment["content"] += " " + sentence["content"]

        # Handle last segment
        if current_segment:
            analysis = self.analyze_segment(
                {"content": current_segment["content"]},
                previous_segment
            )
            current_segment["topic"] = analysis["topic"]
            current_segment["keywords"] = analysis["keywords"]
            segments.append(current_segment)

        print(f"Identified {len(segments)} segments.")
        return segments

def process_transcript(transcript: List[Dict], project_path: str, num_topics: int = 5) -> Dict:
    """Process transcript for topic modeling and segmentation."""
    analyzer = TopicAnalyzer()
    
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
