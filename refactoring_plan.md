# Video Topic Splitter Refactoring Plan

## Overview
Refactor the video frame extraction and analysis system to better integrate with transcript timestamps and improve topic modeling accuracy.

## Current Architecture
1. Topic Modeling
   - Uses OpenRouter/phi-4 for topic analysis
   - Processes transcript in batches
   - Identifies segment boundaries based on topic shifts

2. Visual Analysis
   - Extracts frames at fixed intervals
   - Performs OCR and logo detection
   - Uses Gemini for frame analysis
   - Operates independently of transcript context

## Proposed Changes

### 1. Frame Extraction Enhancement
- Modify `split_and_analyze_video()` to extract frames at:
  - Segment boundaries from transcript
  - Key points within segments (start, middle, end)
  - Scene change detection points
- Store frames with timestamp metadata
- Implement frame deduplication

### 2. Context-Aware Analysis
- Enhance `analyze_frame_for_software()` to include:
  - Transcript context from surrounding timestamps
  - Topic information from segment
  - Previous frame analysis results
- Update Gemini prompts to incorporate transcript context

### 3. Integration Points
- Add frame extraction during topic modeling phase
- Create new class `ContextualFrameAnalyzer` to handle:
  - Frame extraction timing
  - Context gathering
  - Analysis coordination

### 4. Implementation Steps

1. Create New Components:
```python
class ContextualFrameAnalyzer:
    def __init__(self, video_path, transcript_segments):
        self.video = VideoFileClip(video_path)
        self.segments = transcript_segments
        
    def extract_segment_frames(self, segment):
        # Extract frames at segment boundaries and key points
        frames = []
        # Implementation details...
        return frames
        
    def analyze_with_context(self, frame, segment_context):
        # Perform analysis with transcript context
        # Implementation details...
        return analysis_result
```

2. Modify Existing Code:
- Update `TopicAnalyzer.identify_segments()` to include frame extraction
- Enhance `analyze_segment_with_gemini()` to use contextual information
- Modify `split_and_analyze_video()` to use new ContextualFrameAnalyzer

3. New Processing Flow:
```
Transcript → Topic Analysis → Frame Extraction → Contextual Analysis → Results
```

### 5. Performance Considerations
- Implement frame caching
- Add parallel processing for frame analysis
- Optimize frame extraction timing
- Add checkpointing for frame analysis results

### 6. Testing Strategy
1. Unit Tests:
   - Frame extraction timing accuracy
   - Context integration
   - Analysis result consistency

2. Integration Tests:
   - End-to-end processing flow
   - Performance benchmarks
   - Memory usage optimization

## Expected Benefits
1. More accurate topic modeling through visual context
2. Better alignment between visual and transcript analysis
3. Improved efficiency in frame processing
4. Enhanced context awareness in analysis results

## Next Steps
1. Create new ContextualFrameAnalyzer class
2. Modify existing analysis components
3. Implement integration points
4. Add tests and benchmarks
5. Document new functionality