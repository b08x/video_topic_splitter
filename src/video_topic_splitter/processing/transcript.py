"""
Transcript processing utilities for video analysis.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

def filter_transcript_by_time_range(
    transcript: List[Dict[str, Any]], 
    start_time: float, 
    end_time: float
) -> List[Dict[str, Any]]:
    """
    Filter transcript items to only include those within a specific time range.
    
    Args:
        transcript: List of transcript dictionaries with 'start_time' and 'end_time' keys
        start_time: Start time in seconds
        end_time: End time in seconds
        
    Returns:
        Filtered list of transcript dictionaries
    """
    filtered_transcript = []
    
    try:
        for item in transcript:
            item_start = float(item.get('start_time', 0))
            item_end = float(item.get('end_time', 0))
            
            # Include items that overlap with the specified range
            if (item_start <= end_time and item_end >= start_time):
                # Optionally adjust timestamps to be relative to the segment
                # adjusted_item = item.copy()
                # adjusted_item['start_time'] = max(0, item_start - start_time)
                # adjusted_item['end_time'] = min(end_time - start_time, item_end - start_time)
                # filtered_transcript.append(adjusted_item)
                
                # Or keep original timestamps
                filtered_transcript.append(item)
                
        logger.debug(f"Filtered transcript from {len(transcript)} to {len(filtered_transcript)} items")
        return filtered_transcript
        
    except Exception as e:
        logger.error(f"Error filtering transcript by time range: {e}", exc_info=True)
        return []

def merge_overlapping_utterances(transcript: List[Dict[str, Any]], max_gap: float = 0.5) -> List[Dict[str, Any]]:
    """
    Merge transcript utterances that are close together or overlapping.
    
    Args:
        transcript: List of transcript dictionaries with 'start_time', 'end_time', and 'text' keys
        max_gap: Maximum gap in seconds between utterances to be merged
        
    Returns:
        List of merged transcript dictionaries
    """
    if not transcript:
        return []
        
    # Sort by start time
    sorted_transcript = sorted(transcript, key=lambda x: float(x.get('start_time', 0)))
    
    merged_transcript = []
    current_item = sorted_transcript[0].copy()
    
    for next_item in sorted_transcript[1:]:
        current_end = float(current_item.get('end_time', 0))
        next_start = float(next_item.get('start_time', 0))
        
        # Check if items should be merged
        if next_start - current_end <= max_gap:
            # Merge items
            current_item['end_time'] = next_item.get('end_time')
            current_item['text'] = f"{current_item.get('text', '')} {next_item.get('text', '')}"
        else:
            # Add current item to results and start a new one
            merged_transcript.append(current_item)
            current_item = next_item.copy()
    
    # Add the last item
    merged_transcript.append(current_item)
    
    logger.debug(f"Merged transcript from {len(transcript)} to {len(merged_transcript)} items")
    return merged_transcript

def extract_speaker_segments(transcript: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract segments where different speakers are talking.
    
    Args:
        transcript: List of transcript dictionaries with 'text' and speaker information
        
    Returns:
        List of speaker segments with start/end times
    """
    if not transcript:
        return []
        
    speaker_segments = []
    current_speaker = None
    segment_start = None
    segment_text = []
    
    for item in transcript:
        # Extract speaker from text or dedicated field if available
        speaker = item.get('speaker')
        
        # If no explicit speaker field, try to extract from text
        if not speaker and 'text' in item:
            text = item.get('text', '')
            # Simple pattern matching - adjust based on your transcript format
            if ': ' in text:
                parts = text.split(': ', 1)
                speaker = parts[0]
        
        item_start = float(item.get('start_time', 0))
        item_end = float(item.get('end_time', 0))
        
        if speaker != current_speaker:
            # Save previous segment if it exists
            if current_speaker and segment_start is not None:
                speaker_segments.append({
                    'speaker': current_speaker,
                    'start_time': segment_start,
                    'end_time': item_start,
                    'text': ' '.join(segment_text)
                })
            
            # Start new segment
            current_speaker = speaker
            segment_start = item_start
            segment_text = [item.get('text', '')]
        else:
            # Continue current segment
            segment_text.append(item.get('text', ''))
    
    # Add the last segment
    if current_speaker and segment_start is not None:
        speaker_segments.append({
            'speaker': current_speaker,
            'start_time': segment_start,
            'end_time': float(transcript[-1].get('end_time', 0)),
            'text': ' '.join(segment_text)
        })
    
    return speaker_segments