# Video Topic Splitter: AI-Powered Video Segmentation and Analysis 🎬

**Automatically segment and analyze videos based on topic changes and visual scenes, leveraging cutting-edge AI.**

This README provides a detailed overview of the Video Topic Splitter, including its functionality, usage, and internal workings. It's based on a thorough analysis of the codebase, so it's a more technical and precise guide than a typical user-oriented README.

## Table of Contents

- [Video Topic Splitter: AI-Powered Video Segmentation and Analysis 🎬](#video-topic-splitter-ai-powered-video-segmentation-and-analysis-)
  - [Table of Contents](#table-of-contents)
  - [About 📖](#about-)
  - [Features ✨](#features-)
  - [Use Cases](#use-cases)
    - [1. IT Technical Support Analysis](#1-it-technical-support-analysis)
    - [2. AI Agent Interaction Analysis](#2-ai-agent-interaction-analysis)
  - [Tech Stack](#tech-stack)
  - [Installation 💾](#installation-)
  - [Usage 💻](#usage-)
    - [Basic Video Processing](#basic-video-processing)
    - [Using a Pre-existing Transcript](#using-a-pre-existing-transcript)
    - [Transcribe Only](#transcribe-only)
    - [Analyze Screenshot](#analyze-screenshot)
    - [Advanced Options](#advanced-options)
  - [Detailed Processing Pipeline (Technical Overview) ⚙️](#detailed-processing-pipeline-technical-overview-️)
    - [1. Initialization and Input Validation](#1-initialization-and-input-validation)
    - [2. YouTube Video Handling (Conditional)](#2-youtube-video-handling-conditional)
    - [3. Transcription (Conditional)](#3-transcription-conditional)
    - [4. Topic Modeling](#4-topic-modeling)
    - [5. Scene-Based Visual Analysis](#5-scene-based-visual-analysis)
    - [6. Final Checkpoint and Results](#6-final-checkpoint-and-results)
  - [Project Structure 📂](#project-structure-)
  - [Configuration ⚙️](#configuration-️)
  - [Dockerized Deployment 🐳](#dockerized-deployment-)
  - [Contributing 🧑‍💻](#contributing-)
  - [License 📜](#license-)

## About 📖

The Video Topic Splitter is a powerful tool designed to automatically segment videos into meaningful sections and provide comprehensive content analysis. It intelligently combines two parallel analysis tracks:

1. **Topic-Based Segmentation:** Analyzes the audio transcript to identify shifts in conversation, creating segments based on distinct topics.
2. **Scene-Based Visual Analysis:** Detects visual scene changes in the video, extracting unique frames for detailed examination.

It utilizes advanced AI models and techniques, including:

- **Speech-to-Text:** Accurate transcription of video audio via OpenAI's Whisper model.
- **Topic Modeling:** Modular identification of distinct topics within the transcribed text using a sophisticated multi-component architecture.
- **Visual Scene Detection:** Automated detection of scene changes using `PySceneDetect`.
- **Unique Frame Extraction:** Intelligent selection of visually unique frames from each scene using `ImageHash` to avoid redundancy.
- **Contextual Frame Analysis:** In-depth analysis of extracted frames using Google's Gemini model.

This makes it ideal for analyzing recordings of technical tutorials, meetings, presentations, support sessions, and more.

## Features ✨

- **Dual Analysis Approach:** Combines topic-based text segmentation with scene-based visual analysis for a comprehensive understanding.
- **Flexible Transcription:**
  - Uses OpenAI's Whisper model for high-quality transcription.
  - Supports custom Whisper API endpoints (e.g., local servers) via the `OPENAI_API_BASE` environment variable.
  - Allows providing an external transcript file (`.srt`, `.vtt`, `.json`) to bypass the transcription step.
- **Advanced Scene Detection:** Employs `PySceneDetect` to accurately identify scene boundaries in the video.
- **Duplicate Frame Prevention:** Uses `ImageHash` to ensure that only visually unique frames from each scene are selected for analysis, improving efficiency.
- **Advanced Topic Modeling (OpenRouter):** Modular topic analysis system with configurable components for intelligent segmentation, caching, and robust JSON parsing.
- **Software Detection (OCR):** Identifies software applications visible in video frames through text extraction.
- **Gemini Analysis (Google):** Offers in-depth summaries and contextual understanding of each video frame.
- **Robust Checkpointing:** Saves progress at each major stage, allowing resumption from interruptions.
- **YouTube URL Support:** Downloads and processes videos directly from YouTube links.

## Use Cases

### 1. IT Technical Support Analysis

- **Error Diagnosis:** Pinpoint the exact moments in a video where errors occur and analyze the surrounding context.
- **Pattern Recognition:** Identify recurring issues and their solutions across multiple support sessions.
- **Knowledge Preservation:** Create a searchable, segmented archive of technical support interactions.
- **Procedural Tracking:** Follow step-by-step troubleshooting procedures and identify deviations.

### 2. AI Agent Interaction Analysis

- **Prompt Engineering Analysis:** Detect effective prompt patterns and how they influence model responses.
- **Model Response Evaluation:** Characterize the quality and relevance of AI model outputs within a specific context.
- **Interaction Pattern Identification:** Understand the flow of conversation between a user and an AI agent.
- **Performance Monitoring:** Assess the overall effectiveness of AI agent interactions over time.

## Tech Stack

- **Transcription:** OpenAI Whisper API (via `curl`).
- **Visual Analysis:** Google's Gemini API.
- **Topic Modeling:** Modular architecture with OpenRouter's `microsoft/phi-4` model, featuring separated concerns for configuration, caching, batching, and response parsing.
- **Scene Detection:** `PySceneDetect`.
- **Image Hashing:** `ImageHash`.
- **Audio Processing:** `ffmpeg`, `ffmpeg-normalize`, `unsilence`.
- **OCR:** `pytesseract` (Tesseract OCR engine).
- **Image/Video Processing:** `opencv-python`, `moviepy`.
- **Core Libraries:** `python-dotenv`, `Pillow`, `yt-dlp`, `scikit-learn`, `nltk`, `progressbar2`.
- **Packaging:** `setuptools`.
- **Runtime:** Python 3.8+.

## Installation 💾

```bash
pip install -r requirements.txt
pip install .
```

## Usage 💻

### Basic Video Processing

This command will perform the full pipeline: audio extraction, transcription, topic modeling, and visual scene analysis.

```bash
video-topic-splitter -i <video_path_or_youtube_url> -o <output_directory>
```

### Using a Pre-existing Transcript

If you have a transcript file, you can skip the audio extraction and transcription steps.

```bash
video-topic-splitter -i <video_path> -o <output_directory> --transcript <path_to_transcript.srt>
```

- `--transcript`: Path to a local transcript file (`.srt`, `.vtt`, or `.json`).

### Transcribe Only

This mode extracts audio (if needed) and generates transcript files (`.json`, `.srt`, `.vtt`) without performing any further analysis.

```bash
video-topic-splitter -i <video_path> -o <output_directory> --transcribe-only
```

### Analyze Screenshot

This mode analyzes a single image file instead of a video.

```bash
video-topic-splitter --analyze-screenshot -i <image_path> -o <output_directory> --screenshot-context "Context for analysis"
```

### Advanced Options

- `--topics <integer>`: The desired number of topics for topic modeling (default: 5).
- `--frames-per-scene <integer>`: Number of unique frames to extract per detected scene (default: 1).
- `--register <it-workflow|gen-ai|tech-support>`: Selects the analysis register for Gemini (default: `it-workflow`).
- `--skip-unsilence`: Disables silence removal during audio preprocessing.
- `--software-list <path_to_text_file>`: Specifies a text file containing a list of software names to detect via OCR.
- `--ocr-lang <language_code>`: Sets the language for OCR (default: `eng`).

## Detailed Processing Pipeline (Technical Overview) ⚙️

The core logic resides in `video_topic_splitter.core.process_video`.

### 1. Initialization and Input Validation

- A project folder is created.
- The tool checks if the input is a valid file path or YouTube URL and if the optional transcript file exists and has a supported format.

### 2. YouTube Video Handling (Conditional)

- If the input is a YouTube URL, `yt-dlp` downloads the video. This step is skipped if a checkpoint indicates it's already complete.

### 3. Transcription (Conditional)

- **If a transcript file is provided (`--transcript`):** The file is parsed, and its content is used for topic modeling.
- **If no transcript is provided:**
  - **Audio Processing:** The audio is extracted, normalized, and optimized for transcription using `ffmpeg` and `unsilence`.
  - **Whisper Transcription:** A `curl` command sends the processed audio to a Whisper API endpoint (configurable via `OPENAI_API_BASE`). The JSON response is parsed into a standard format.
- The resulting transcript is saved in `.json`, `.srt`, and `.vtt` formats.

### 4. Topic Modeling

- **Modular Architecture:** The topic modeling system has been refactored into focused components:
  - **TopicAnalyzerConfig:** Centralized configuration management with validation
  - **ResponseParser:** Robust JSON parsing with multiple fallback strategies
  - **AsyncCache:** Thread-safe async caching with LRU eviction
  - **SegmentBatcher:** Smart batching logic with similarity-based boundary detection
  - **TopicAnalyzer:** Core analysis orchestration using OpenRouter's `microsoft/phi-4` model
- **Enhanced Features:** Configurable parameters, debug logging, progress tracking, and improved error handling
- **Output:** Generates segment metadata with start/end times, dominant topics, keywords, and confidence scores

### 5. Scene-Based Visual Analysis

- **Scene Detection:** `PySceneDetect` is used to detect scene changes in the video, creating a list of scenes.
- **Unique Frame Extraction:** For each detected scene, a specified number of frames (`--frames-per-scene`) are extracted.
- **Duplicate Filtering:** `ImageHash` calculates a perceptual hash for each extracted frame. Frames that are too visually similar to already selected frames are discarded to ensure uniqueness.
- **Frame Analysis:** Each unique frame is analyzed:
  - **OCR:** `pytesseract` detects software names.
  - **Gemini Analysis:** Google's Gemini model provides a textual description of the visual content.

### 6. Final Checkpoint and Results

- The final results, including topics, text-based segments, and scene-based visual analyses, are saved to `results.json`.
- A final checkpoint is saved to mark the process as complete.

## Project Structure 📂

### Code Architecture

The topic modeling system has been refactored into a modular architecture:

```
src/video_topic_splitter/analysis/
├── topic_modeling.py          # Main interface and orchestration
├── topic_analyzer_config.py   # Configuration management with validation
├── response_parser.py         # JSON parsing with multiple fallback strategies
├── async_cache.py            # Thread-safe async caching with LRU eviction
├── segment_batcher.py        # Smart batching with similarity analysis
└── topic_analyzer.py         # Core analysis engine with OpenRouter integration
```

**Key Benefits:**
- **Maintainability:** Each module has a single responsibility
- **Testability:** Components can be tested independently
- **Extensibility:** Easy to add new parsers or cache implementations
- **Configuration-Driven:** Centralized configuration with validation

### Output Directory Structure

The tool creates a project directory with the following structure:

```
<output_directory>/
└── <project_name>_<timestamp>/
    ├── audio/
    │   └── mono_resampled_audio.m4a  (Processed audio for transcription)
    ├── scenes/
    │   ├── frames/
    │   │   └── Scene-001-01.jpg     (Extracted frames)
    │   └── scenes.csv               (Scene list from PySceneDetect)
    ├── transcription.json           (Raw Whisper API response)
    ├── transcript.json              (Processed transcript)
    ├── transcript.srt               (SRT subtitle file)
    ├── transcript.vtt               (VTT subtitle file)
    ├── results.json                 (Final combined analysis results)
    └── checkpoint.pkl               (Checkpoint file for resuming)
```

## Configuration ⚙️

- **API Keys:** You *must* set the following environment variables. A `.env` file is recommended.
  - `OPENAI_API_KEY`: Your OpenAI API key. This is optional if you are using a local Whisper server that does not require authentication.
  - `OPENAI_API_BASE`: (Optional) The base URL for the Whisper API. Defaults to `https://api.openai.com/v1`. Use this to point to a local inference server (e.g., `http://localhost:8080/v1`).
  - `GEMINI_API_KEY`: Your Google Gemini API key.
  - `OPENROUTER_API_KEY`: Your OpenRouter API key.

    ```bash
    # .env file
    OPENAI_API_KEY=your_openai_key
    OPENAI_API_BASE=http://localhost:8080/v1
    GEMINI_API_KEY=your_gemini_key
    OPENROUTER_API_KEY=your_openrouter_key
    ```

## Dockerized Deployment 🐳

The project includes a `Dockerfile` and `docker-compose.yml` for containerized deployment. See the files for detailed setup instructions.

## Contributing 🧑‍💻

This project was generated as an exercise in utilizing Large Language Models to develop a Python application.

## License 📜

[MIT License](LICENSE)
