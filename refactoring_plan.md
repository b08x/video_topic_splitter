# Visual Context Integration Plan

## Overview
Integrate visual information from video frames into the topic analysis process by extracting and analyzing representative screenshots from each transcript segment before topic modeling.

## Current Architecture
- `ContextualFrameAnalyzer`: Handles frame extraction and analysis with transcript context
- `TopicAnalyzer`: Performs topic modeling on transcript segments
- `visual_analysis.py`: Contains visual analysis functionality

## Implementation Plan

### 1. Frame Extraction Enhancement
- Leverage existing `ContextualFrameAnalyzer.extract_segment_frames()` method
- Already extracts frames at:
  - Segment start
  - Segment end
  - Internal frames at evenly spaced intervals
- Add frame quality assessment to select most representative frames

### 2. Visual Feature Integration
- Create new class `VisualFeatureExtractor` in visual_analysis.py
- Extract visual features from frames:
  - Software UI elements (existing OCR and logo detection)
  - Scene composition
  - Text content
  - Visual elements (charts, diagrams, code blocks)
- Generate visual context descriptors

### 3. Topic Modeling Enhancement
Modify `TopicAnalyzer` to incorporate visual information:
1. Extract frames before topic analysis
2. Generate visual context for each segment
3. Combine visual and transcript features for topic modeling
4. Update topic confidence scoring to include visual relevance

### 4. Data Flow Changes
1. Video input → Frame extraction
2. Frame analysis and feature extraction
3. Visual context generation
4. Combined topic modeling (text + visual)
5. Topic assignment with visual context

### 5. Screenshot Management

#### Storage Structure
- Create a screenshots directory in the project output:
  ```
  project_dir/
  ├── results.json
  ├── screenshots/
  │   ├── segment_1/
  │   │   ├── start.jpg
  │   │   ├── internal_1.jpg
  │   │   ├── internal_2.jpg
  │   │   └── end.jpg
  │   ├── segment_2/
  │   │   └── ...
  ```
- Organize screenshots by segment
- Use consistent naming convention
- Store metadata (timestamp, frame type) in results.json

#### Screenshot Handling
1. Frame Selection:
   - Quality assessment to choose best frames
   - Avoid duplicate/similar frames
   - Maintain reasonable storage size

2. Storage Management:
   - Configurable image format (jpg/png)
   - Adjustable quality settings
   - Optional compression
   - Cleanup of temporary frames

3. Access Interface:
   - Methods to retrieve frames by segment
   - Utilities for frame metadata
   - Integration with web interface

### 6. Implementation Steps

#### Phase 1: Frame Processing
1. Update `ContextualFrameAnalyzer`:
   - Add frame quality assessment
   - Optimize frame selection
   - Add visual feature extraction

2. Create `VisualFeatureExtractor`:
   - Implement visual feature extraction
   - Add context generation methods
   - Integrate with existing analysis tools

#### Phase 2: Topic Analysis Integration
1. Modify `TopicAnalyzer`:
   - Add visual context handling
   - Update topic modeling to use combined features
   - Enhance topic confidence scoring

2. Update `process_transcript()`:
   - Add frame extraction step
   - Integrate visual context
   - Modify topic analysis flow

#### Phase 3: Results Enhancement
1. Update results structure to include:
   - Representative screenshots
   - Visual context descriptions
   - Combined topic analysis results

2. Modify checkpoint system to handle:
   - Frame extraction progress
   - Visual analysis state
   - Combined analysis results

### 6. Technical Considerations
- Frame storage and management:
  - Disk space requirements
  - Image format optimization
  - Metadata storage efficiency
  - Access patterns and caching
- Processing performance optimization:
  - Parallel frame extraction
  - Batch processing capabilities
  - Resource usage monitoring
- Memory usage for frame analysis:
  - Frame buffer management
  - Cleanup of processed frames
  - Memory-efficient analysis
- Error handling:
  - Frame extraction failures
  - Storage write errors
  - Corrupt frame detection
  - Recovery procedures
- Checkpoint system updates:
  - Screenshot progress tracking
  - Resume capability for long videos
  - Partial results handling

### 7. Testing Strategy
1. Unit Tests:
   - Frame extraction and selection
   - Visual feature extraction
   - Combined topic analysis
   - Screenshot storage and retrieval
   - Image quality assessment
   - Metadata handling

2. Integration Tests:
   - End-to-end workflow
   - Visual context integration
   - Results format validation
   - Screenshot organization
   - Storage structure creation
   - Access interface functionality

3. Performance Tests:
   - Memory usage
   - Processing time
   - Storage requirements
   - I/O performance
   - Compression efficiency
   - Parallel processing effectiveness

4. Storage Tests:
   - Disk space utilization
   - File organization
   - Metadata consistency
   - Access patterns
   - Cleanup procedures

## Success Criteria
1. Representative frames extracted for each segment
2. Visual context successfully integrated into topic analysis
3. Improved topic relevance with visual information
4. Efficient processing and storage usage
5. Robust error handling and recovery