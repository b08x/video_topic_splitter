# Video Topic Splitter 🎬

A powerful AI-driven tool for intelligent video content analysis and segmentation.

## Table of Contents

- [Video Topic Splitter 🎬](#video-topic-splitter-)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [System Architecture](#system-architecture)
  - [Tech Stack](#tech-stack)
  - [Development](#development)
  - [Project Overview](#project-overview)
  - [Use Cases](#use-cases)
    - [1. IT Technical Support Analysis](#1-it-technical-support-analysis)
    - [2. AI Agent Interaction Analysis](#2-ai-agent-interaction-analysis)
  - [Design Philosophy](#design-philosophy)
  - [Limitations](#limitations)
  - [Contributing](#contributing)
  - [License](#license)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/video-topic-splitter.git
cd video-topic-splitter

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
```

## Quick Start

Basic usage:
```bash
# Process a video file
video-topic-splitter -i your_video.mp4 -o output_segments

# Analyze with specific workflow
video-topic-splitter -i tech_talk.mp4 -o tech_insights --register it-workflow
```

## System Architecture

The system processes videos through multiple specialized components working in sequence:

```mermaid
sequenceDiagram
    actor User
    participant CLI
    participant YTD as YouTube Downloader
    participant AP as Audio Processor
    participant TS as Transcription Service
    participant TA as Topic Analyzer
    participant VA as Video Analyzer
    participant TM as Thumbnail Manager

    User->>CLI: Initiate process with options
    activate CLI

    alt YouTube URL provided
        CLI->>YTD: Download video
        YTD-->>CLI: Video file
    end

    CLI->>AP: Process audio
    AP->>AP: Extract audio
    AP->>AP: Normalize audio
    opt If not skipped
        AP->>AP: Remove silence
    end
    AP-->>CLI: Processed audio

    CLI->>TS: Request transcription
    TS-->>CLI: Transcript

    CLI->>TA: Analyze topics
    TA->>TA: Perform LDA analysis
    TA->>TA: Identify segments
    TA-->>CLI: Topic analysis results

    CLI->>VA: Analyze video content
    VA->>VA: Detect software logos
    VA->>VA: Perform OCR on frames
    VA->>VA: Analyze frame content
    VA-->>CLI: Video analysis results

    CLI->>TM: Generate thumbnails
    TM->>TM: Extract key frames
    TM->>TM: Create thumbnails
    TM-->>CLI: Thumbnail set

    CLI-->>User: Complete analysis results
    deactivate CLI
```

## Tech Stack

- **Transcription**: Deepgram & Groq APIs for speech-to-text
- **Visual Analysis**: Google's Gemini for frame analysis
- **Topic Modeling**: OpenRouter's phi-4 model
- **Audio Processing**: FFmpeg for audio extraction and normalization
- **OCR**: Tesseract for text extraction from frames
- **Core**: Python with extensive use of async/await for performance

## Development

Support workflow integration:

```mermaid
flowchart TD
    A[Support Request Received] --> B{Is Video Analysis Required?}
    B -->|No| C[Standard Support Process]
    B -->|Yes| D[Initiate Video Analysis]
    D --> E[Run Video Processing Sequence]
    
    subgraph "Video Processing Sequence"
    E1[Download Video if URL] --> E2[Process Audio]
    E2 --> E3[Transcribe Audio]
    E3 --> E4[Analyze Topics]
    E4 --> E5[Analyze Video Content]
    E5 --> E6[Generate Thumbnails]
    end
    
    E --> F[Review Analysis Results]
    F --> G{Is Further Action Needed?}
    G -->|Yes| H[Assign to Specialist]
    G -->|No| I[Update Knowledge Base]
    H --> J[Resolve Issue]
    I --> K[Close Support Ticket]
    J --> K
    C --> K
```

## Project Overview

Video Topic Splitter is designed to transform complex video content into structured, navigable segments. It combines multiple AI technologies to understand and analyze multimedia content effectively, making it particularly valuable for technical content and educational materials.

The system provides:
- Accurate speech-to-text conversion
- Intelligent topic segmentation
- Visual content analysis
- Automated thumbnail generation
- Structured output for knowledge base integration

## Use Cases

### 1. IT Technical Support Analysis

- Detailed error diagnosis and pattern recognition
- Solution trajectory mapping
- Procedural knowledge preservation
- Diagnostic step tracking

### 2. AI Agent Interaction Analysis

- Prompt engineering pattern detection
- Model response characterization
- Interaction pattern analysis
- Performance evaluation

## Design Philosophy

The Video Topic Splitter approaches multimedia analysis through three core principles:

1. **Intelligent Segmentation**: Content is analyzed not just for transitions, but for meaningful topic boundaries and context shifts.

2. **Multi-Modal Understanding**: Combines audio transcription, visual analysis, and topic modeling for comprehensive content understanding.

3. **Knowledge Transformation**: Transforms raw video content into structured, actionable documentation and insights.

## Limitations

Current limitations of the system:

- Complex technical discussions may challenge topic boundary detection
- Visual analysis accuracy depends on video quality
- Processing time scales with video length
- Some specialized technical terminology may require domain-specific training

Version: 0.3

## Contributing

Contributions are welcome! Please feel free to submit pull requests with improvements or bug fixes.

## License

MIT License - See LICENSE file for details.

---

_Built with precision for technical content analysis_
