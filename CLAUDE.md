# Video Topic Splitter - Claude Code Guide

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Project Overview

The Video Topic Splitter is a sophisticated Python-based AI tool that automatically segments videos based on topic analysis and visual scene detection. The tool combines multiple AI services to create an organized, topic-based video analysis system.

**Current Version**: 0.2.0 (Major refactoring with organized directory structure)

### Core AI Components
- **Audio Transcription**: OpenAI Whisper API with configurable endpoints
- **Topic Modeling**: OpenRouter's microsoft/phi-4 model for topic segmentation
- **Enhanced NLP Analysis**: spaCy for sophisticated transcript analysis including tokenization, lemmatization, NER, and dependency parsing
- **Visual Analysis**: Google Gemini API for frame-level content analysis
- **Scene Detection**: PySceneDetect for visual scene boundary detection
- **Audio Processing**: ffmpeg, ffmpeg-normalize, unsilence for audio manipulation
- **OCR**: Tesseract (pytesseract) for text extraction from video frames

### Architecture Overview

The project uses a modular architecture with these key components:

1. **Core Processing** (`core.py`): Main orchestration with checkpointing
2. **Project Structure** (`project_structure.py`): Organized output directory management
3. **Video Segmentation** (`processing/video/video_segmentation.py`): FFmpeg-based video/audio cutting
4. **Multimodal Analysis** (`analysis/multimodal_analysis.py`): Comprehensive segment analysis
5. **Progress Tracking** (`progress_tracker.py`): Hierarchical progress reporting
6. **Segment Analysis** (`analysis/segment_analysis.py`): Per-segment processing orchestration

### Key Processing Pipeline

1. **Audio Processing**: Extract, normalize, remove silence
2. **Transcription**: Whisper API transcription with retry logic
3. **Topic Modeling**: Identify topic segments using OpenRouter
4. **Video Segmentation**: Split video into topic-based segments with FFmpeg
5. **Multimodal Analysis**: Analyze each segment (transcript + visual + audio)
6. **Results Consolidation**: Generate organized output structure

## Quick Start Commands

### Installation
```bash
pip install -r requirements.txt
pip install .
```

### Basic Usage
```bash
# Process a video file
video-topic-splitter -i video.mp4 -o output_dir

# Process YouTube URL
video-topic-splitter -i "https://youtube.com/watch?v=VIDEO_ID" -o output_dir

# Use existing transcript
video-topic-splitter -i video.mp4 -o output_dir --transcript transcript.srt

# Transcribe only (no analysis)
video-topic-splitter -i video.mp4 -o output_dir --transcribe-only

# Screenshot analysis
video-topic-splitter -i screenshot.png -o output_dir --analyze-screenshot

# With software detection
video-topic-splitter -i video.mp4 -o output_dir --software-list software.txt
```

### Development Commands
```bash
# Run with development settings
python -m video_topic_splitter.cli -i video.mp4 -o output_dir

# Run tests (check for test files first)
python -m pytest tests/

# Check code style
python -m flake8 src/
```

### Testing
The project uses checkpointing, so you can test incrementally:
1. Run with `--transcribe-only` to test transcription
2. Process a short video segment first
3. Use existing transcript to test analysis components
4. Check progress JSON output for debugging

## Configuration

### Required Environment Variables
```bash
# .env file
OPENAI_API_KEY=your_openai_key
OPENAI_API_BASE=https://api.openai.com/v1  # or local server
GEMINI_API_KEY=your_gemini_key
OPENROUTER_API_KEY=your_openrouter_key
```

### Common Options
- `--topics N`: Number of topics for modeling (default: 5)
- `--frames-per-scene N`: Frames to extract per scene (default: 1)
- `--register TYPE`: Analysis register (it-workflow, gen-ai, tech-support)
- `--skip-unsilence`: Skip silence removal
- `--progress-json`: Output progress in JSON format

## Output Directory Structure

```
output/
└── project_name_timestamp/
    ├── input/
    │   ├── source_video.mp4
    │   └── source_transcript.srt
    ├── transcript/
    │   ├── transcript.json
    │   ├── transcript.srt
    │   └── transcript.vtt
    ├── topic_segments/
    │   ├── topic_1_segment_name/
    │   │   ├── segment_001.mp4
    │   │   ├── segment_001.m4a
    │   │   ├── frames/
    │   │   ├── multimodal_analysis.json
    │   │   ├── speaker_transcript.json
    │   │   └── segment_summary.json
    │   └── topic_2_segment_name/
    ├── final_analysis/
    │   └── timeline.json
    ├── results.json
    └── checkpoint.pkl
```

## Key Classes and Modules

### Core Classes
- `process_video()` in `core.py`: Main processing pipeline entry point
- `ProjectStructure`: Manages organized output directory structure
- `ProgressTracker`: Hierarchical progress reporting system
- `SegmentProcessor`: Orchestrates per-segment multimodal analysis
- `MultimodalAnalyzer`: Combines transcript, visual, and audio analysis

### Processing Modules
- `video_segmentation.py`: FFmpeg-based video/audio cutting with keyframe alignment
- `multimodal_analysis.py`: Comprehensive segment analysis combining all modalities
- `segment_analysis.py`: Per-segment processing orchestration
- `topic_modeling.py`: OpenRouter integration for topic analysis
- `visual_analysis.py`: Scene detection and frame analysis (legacy compatibility)

## Important Implementation Details

### Error Handling
- Comprehensive try/catch blocks with logging throughout
- Retry logic for API calls with exponential backoff
- Graceful degradation when components fail

### API Integration
- **OpenRouter**: JSON parsing with multiple fallback strategies for robustness
- **Gemini**: Image analysis with PIL integration for frame processing
- **Whisper**: Audio transcription with configurable endpoints (supports local servers)

### Performance Optimizations
- Checkpointing system for resumable processing after interruptions
- Progress tracking with ETA calculations and real-time updates
- Efficient frame extraction using ImageHash deduplication to avoid redundant processing

### Dependencies
Key external tools required:
- `ffmpeg` (video/audio processing)
- `ffmpeg-normalize` (audio normalization) 
- `unsilence` (silence removal)
- `tesseract` (OCR functionality)

### Python Dependencies
Key Python packages:
- `spacy>=3.7.0` (advanced NLP analysis)
- `scipy` (for semantic similarity calculations)
- Required spaCy language model: `en_core_web_md` (includes word vectors for semantic analysis)

Install spaCy model:
```bash
python -m spacy download en_core_web_md
```

## Common Issues & Solutions

### Audio Processing Issues
- Ensure ffmpeg is properly installed and in PATH
- Check audio codec compatibility for input files
- Verify file permissions for input/output directories

### API Rate Limiting
- Implement proper retry logic with exponential backoff
- Monitor API quota usage across providers
- Use local inference servers when possible

### Memory Issues
- Process large videos in segments to manage memory usage
- Clear intermediate files when processing completes
- Monitor memory usage during long processing sessions

## Debugging Tips

- Check `checkpoint.pkl` for processing state and resumption points
- Review `results.json` for final outputs and error summaries
- Monitor console output for real-time progress updates
- Use `--progress-json` for programmatic progress monitoring
- Examine segment-level analysis files for detailed insights

## Code Style & Development

- Follow PEP 8 conventions for Python code
- Use type hints for function signatures where possible
- Add comprehensive docstrings for classes and methods
- Log important operations and errors with appropriate levels
- Use meaningful variable names and clear code structure

## Enhanced Transcript Analysis with spaCy

The transcript analysis has been significantly upgraded using spaCy's advanced NLP capabilities:

### Key Features
- **Superior Tokenization**: Handles contractions, punctuation, and abbreviations correctly (vs. basic `text.split()`)
- **Lemmatization**: Groups word variations ("analyze", "analyzing", "analyzed") under base forms
- **Stop Word Removal**: Filters out common words using linguistic understanding, not length heuristics
- **Noun Chunking**: Extracts meaningful multi-word phrases like "quarterly earnings report"
- **Named Entity Recognition**: Automatically identifies people, organizations, dates, money, locations
- **Part-of-Speech Analysis**: Categorizes words by grammatical function for content filtering
- **Dependency Parsing**: Extracts subject-verb-object relationships showing "who did what"
- **Semantic Similarity**: Enables topic clustering and semantic search (with word vectors)

### Output Enhancements
The enhanced analysis provides structured data including:
- **Named Entities**: `{"PERSON": ["John Smith"], "ORG": ["Acme Corp"], "MONEY": ["$50,000"]}`
- **Key Actions**: Subject-verb-object triplets showing main actions and actors
- **Technical Elements**: Automatically detected software, tools, and technical terms
- **Linguistic Features**: Part-of-speech distributions and grammatical relationships
- **Semantic Insights**: Document coherence and similarity scores

### Backward Compatibility
The system maintains full backward compatibility while adding enhanced analysis as optional structured data under `enhanced_analysis` key.

## Recent Changes (v0.2.0)

- **Major Refactoring**: Added organized directory structure with topic-based segments
- **Enhanced NLP**: Integrated spaCy for sophisticated transcript analysis with NER, lemmatization, and dependency parsing
- **Video Segmentation**: Implemented FFmpeg-based video/audio cutting with keyframe alignment
- **Multimodal Analysis**: Created comprehensive segment analysis pipeline
- **Progress Tracking**: Enhanced progress system with hierarchical reporting
- **Error Handling**: Improved error handling and retry logic throughout
- **Logging**: Added comprehensive logging across all components