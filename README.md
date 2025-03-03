# Video Topic Splitter: AI-Powered Video Segmentation and Analysis 🎬

**Automatically segment and analyze videos based on topic changes, leveraging cutting-edge AI.**

This README provides a detailed overview of the Video Topic Splitter, including its functionality, usage, and internal workings.  It's based on a thorough analysis of the codebase, so it's a more technical and precise guide than a typical user-oriented README.

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
    - [Transcribe Only](#transcribe-only)
    - [Analyze Screenshot](#analyze-screenshot)
    - [Advanced Options](#advanced-options)
  - [Detailed Processing Pipeline (Technical Overview) ⚙️](#detailed-processing-pipeline-technical-overview-️)
    - [1. Initialization and Checkpoint Loading](#1-initialization-and-checkpoint-loading)
    - [2. YouTube Video Handling (Conditional)](#2-youtube-video-handling-conditional)
    - [3. Audio Processing (`handle_audio_video`)](#3-audio-processing-handle_audio_video)
    - [4. Transcription and Analysis (`handle_transcription` or Transcribe-Only)](#4-transcription-and-analysis-handle_transcription-or-transcribe-only)
    - [5. Final Checkpoint and Results](#5-final-checkpoint-and-results)
    - [Screenshot Analysis Workflow (Separate Path)](#screenshot-analysis-workflow-separate-path)
  - [Project Structure 📂](#project-structure-)
  - [Configuration ⚙️](#configuration-️)
  - [Dockerized Deployment 🐳](#dockerized-deployment-)
    - [Docker Compose Setup](#docker-compose-setup)
    - [Dockerfile Explanation](#dockerfile-explanation)
  - [Contributing 🧑‍💻](#contributing-)
  - [License 📜](#license-)

## About 📖

The Video Topic Splitter is a powerful tool designed to automatically segment videos into meaningful sections based on topic shifts and provide comprehensive content analysis. It utilizes advanced AI models and techniques, including:

- **Speech-to-Text:**  Accurate transcription of video audio.
- **Topic Modeling:**  Identification of distinct topics within the transcribed text.
- **Visual Analysis:**  Extraction of visual information (text and logos) and contextual analysis of video frames.
- **Intelligent Segmentation:**  Division of the video into coherent segments based on identified topic changes.

This makes it ideal for analyzing recordings of technical tutorials, meetings, presentations, support sessions, and more.

## Features ✨

- **Automatic Video Segmentation:**  Divides videos into topic-based segments without manual intervention.
- **Topic Modeling (OpenRouter's phi-4):**  Identifies the primary topic discussed in each segment.
- **Transcription (Deepgram or Groq API):**  Provides high-quality transcripts of the video audio.
- **Software Detection (OCR and Logo Recognition):**  Identifies software applications visible in the video through text extraction and logo matching.
- **Gemini Analysis (Google's Gemini API):**  Offers in-depth summaries and contextual understanding of each video segment, tailored to a specific register (IT Workflow, Generative AI, or Tech Support).
- **Robust Checkpointing:**  Saves progress and allows resuming from interruptions, ensuring no data loss.
- **YouTube URL Support:**  Downloads and processes videos directly from YouTube links.
- **Customizable Analysis Registers:**  Tailor the analysis to specific domains (IT, AI, support) for more relevant insights.
- **Screenshot Analysis Mode:**  Analyzes individual screenshots for software and content, separate from full video processing.

## Use Cases

### 1. IT Technical Support Analysis

- **Error Diagnosis:** Pinpoint the exact moments in a video where errors occur and analyze the surrounding context.
- **Pattern Recognition:** Identify recurring issues and their solutions across multiple support sessions.
- **Knowledge Preservation:**  Create a searchable, segmented archive of technical support interactions.
- **Procedural Tracking:**  Follow step-by-step troubleshooting procedures and identify deviations.

### 2. AI Agent Interaction Analysis

- **Prompt Engineering Analysis:**  Detect effective prompt patterns and how they influence model responses.
- **Model Response Evaluation:**  Characterize the quality and relevance of AI model outputs within a specific context.
- **Interaction Pattern Identification:**  Understand the flow of conversation between a user and an AI agent.
- **Performance Monitoring:**  Assess the overall effectiveness of AI agent interactions over time.

## Tech Stack

- **Transcription:**  Deepgram API (default) or Groq API (optional) for speech-to-text.
- **Visual Analysis:**  Google's Gemini API for frame-level content analysis.
- **Topic Modeling:**  OpenRouter's `microsoft/phi-4` model for topic identification and segmentation.
- **Audio Processing:**  `ffmpeg` and `ffmpeg-normalize` for audio extraction, conversion, normalization, and dynamic range compression; `unsilence` for optional silence removal.
- **OCR:**  `pytesseract` (Tesseract OCR engine) for text extraction from video frames.
- **Image/Video Processing:**  `opencv-python` for frame manipulation, logo detection, and quality assessment; `moviepy` for video loading and audio extraction.
- **Core Libraries:**  `python-dotenv`, `groq`, `openai`, `google-generativeai`, `videogrep`, `scikit-learn`, `nltk`, `progressbar2`, `Pillow`, `yt-dlp`.
- **Concurrency:**  `asyncio` for asynchronous API calls (e.g., OpenRouter), improving performance.
- **Packaging:** `setuptools`
- **Runtime:** Python 3.8+

## Installation 💾

```bash
pip install video_topic_splitter
```

## Usage 💻

### Basic Video Processing

```bash
video-topic-splitter -i <input_video_path_or_youtube_url> -o <output_directory> --topics <number_of_topics> --register <register>
```

- `-i`:  Path to a local video file (MP4, MKV) or a YouTube video URL.
- `-o`:  Base directory where the project folder will be created.
- `--topics`:  The desired number of topics for topic modeling (default: 5).
- `--register`:  Select analysis register (default: it-workflow). Options: `it-workflow`, `gen-ai`, `tech-support`

**Example (YouTube URL):**

```bash
video-topic-splitter -i "https://www.youtube.com/watch?v=dQw4w9WgXcQ" -o output --topics 5 --register it-workflow
```

**Example (Local Video File):**

```bash
video-topic-splitter -i my_video.mp4 -o output --topics 3 --register gen-ai
```

### Transcribe Only

This mode generates a transcript but skips topic modeling and visual analysis.

```bash
video-topic-splitter -i <input_video_path> -o <output_directory> --transcribe-only
```

### Analyze Screenshot

This mode analyzes a single image file.

```bash
video-topic-splitter -i <image_path> -o <output_directory> --analyze-screenshot --screenshot-context "Context for analysis" --software-list software.txt --logo-db logos/
```

- `--screenshot-context`: optional context to consider.
- `--software-list`: text file containing software to detect (one per line)
- `--logo-db`: path to logo database directory.

### Advanced Options

- `--api <deepgram|groq>`:  Selects the transcription API (default: `deepgram`).
- `--skip-unsilence`:  Disables silence removal during audio preprocessing.
- `--software-list <path_to_text_file>`:  Specifies a text file containing a list of software names to detect (one software name per line).
- `--logo-db <path_to_logo_directory>`:  Provides a directory containing logo images (PNG format) for software detection.  Logo file names should match the software names (e.g., `firefox.png`, `vscode.png`).
- `--ocr-lang <language_code>`:  Sets the language for OCR (default: `eng` for English).  Use Tesseract language codes (e.g., `fra` for French, `spa` for Spanish).
- `--logo-threshold <float_value>`:  Adjusts the confidence threshold for logo detection (0.0 to 1.0, default: 0.8).  Higher values are stricter.
- `--thumbnail-interval <seconds>`:  Sets the time interval (in seconds) between generated thumbnails (default: 5).
- `--max-thumbnails <integer>`:  Limits the maximum number of thumbnails generated per segment (default: 5).
- `--min-thumbnail-confidence <float_value>`: The minimum confidence for thumbnail analysis.

## Detailed Processing Pipeline (Technical Overview) ⚙️

The core processing logic resides in `video_topic_splitter.core.process_video`. Here's a breakdown of the steps:

### 1. Initialization and Checkpoint Loading

- The tool loads a checkpoint file (`checkpoint.pkl`) from the project directory to resume processing if it exists. This allows the tool to pick up where it left off if interrupted.

### 2. YouTube Video Handling (Conditional)

- If the input is a YouTube URL, the tool uses `yt-dlp` to download the video in the best available MP4 format and save it to the project directory (`source_video.mp4`). The best quality thumbnail is also downloaded.
- This step is skipped if the checkpoint indicates it has already been completed.

### 3. Audio Processing (`handle_audio_video`)

- **Normalization:** The audio levels of the video are normalized using `ffmpeg-normalize`.
- **Silence Removal (Optional):** If `--skip-unsilence` is not used, silent parts of the audio are sped up using the `unsilence` library.
- **Audio Extraction:** The audio stream is extracted from the (potentially unsilenced) video using `moviepy` and saved as an Opus file (`extracted_audio.opus`).
- **Mono Conversion and Resampling:** The extracted audio is converted to mono, resampled to 16kHz, and encoded as AAC (`mono_resampled_audio.m4a`) using `ffmpeg`. Volume adjustment, high-pass filtering, and dynamic range compression are also applied to optimize for transcription.
- Checkpointing is used to avoid reprocessing audio if these steps have already been completed.

### 4. Transcription and Analysis (`handle_transcription` or Transcribe-Only)

- **Transcription (Deepgram or Groq):** The processed audio (`mono_resampled_audio.m4a`) is transcribed using either the Deepgram API (default) or the Groq API.  The raw transcription and a processed version (segmented into sentences with timestamps) are saved to JSON files (`transcription.json` and `transcript.json`).
- **Topic Modeling (OpenRouter's phi-4):** If `--transcribe-only` is *not* used, the `TopicAnalyzer` class (in `video_topic_splitter.analysis.topic_modeling.py`) analyzes the transcript using OpenRouter's `microsoft/phi-4` model.  It identifies topic shifts and generates segment metadata (start/end times, dominant topic, keywords).  This uses TF-IDF similarity and a configurable threshold to detect topic changes.  Asynchronous calls to the OpenRouter API are used for performance.
- **Visual Analysis (`split_and_analyze_video`):** The video is split into segments based on the topic boundaries. For each segment:
  - Key frames are extracted (start, end, and a configurable number of internal frames).
  - Frame quality is assessed.
  - Software logos are detected using template matching with OpenCV (`detect_software_logos`).
  - OCR is performed using `pytesseract` to detect text (`detect_software_names`).
  - Google's Gemini API (`analyze_with_gemini`) analyzes each frame, providing a textual description of the visual content in the context of the segment's transcript and identified topic.
  - Screenshots of high-quality key frames are saved.
  - A visual summary is generated for each segment.
- Checkpointing occurs after transcription, topic modeling, and visual analysis to enable resuming from each stage.

### 5. Final Checkpoint and Results

- The final results (topics, segment metadata, visual analysis summaries) are saved to `results.json`.
- The checkpoint is updated to reflect the completion of the entire process.

### Screenshot Analysis Workflow (Separate Path)

- If the `--analyze-screenshot` flag is provided, the tool skips the video processing steps and instead analyzes a single image file.
- It performs software detection (OCR and logo matching) and uses the Gemini API to analyze the screenshot content.
- Results are saved to `results.json`.

## Project Structure 📂

The tool creates a project directory for each video processed. The structure is as follows:

```
<output_directory>/
└── <project_name>_<timestamp>/  (e.g., my_video_20240315_143000)
    ├── audio/
    │   ├── extracted_audio.opus       (Raw extracted audio)
    │   └── mono_resampled_audio.m4a  (Processed audio for transcription)
    ├── segments/
    │   ├── segment_1/               (Individual segment directories)
    │   │    └── ...
    │   └── analyzed_segments.json   (Visual analysis results for all segments)
    ├── thumbnails/
    │   ├── metadata.json            (Metadata about generated thumbnails)
    │   └── thumbnail_001.jpg        (Thumbnail images)
    ├── transcription.json          (Raw transcription data from Deepgram/Groq)
    ├── transcript.json              (Processed transcript - sentence segmentation)
    ├── results.json                (Final results: topics, segments, analysis)
    └── checkpoint.pkl              (Checkpoint file for resuming)
```

## Configuration ⚙️

- **API Keys:** You *must* set the following environment variables with your API keys:
  - `DG_API_KEY`: Your Deepgram API key.
  - `GROQ_API_KEY`: Your Groq API key (if using the `--api groq` option).
  - `GEMINI_API_KEY`: Your Google Gemini API key.
  - `OPENROUTER_API_KEY`: Your OpenRouter API key.

    You can set these in your shell or use a `.env` file (recommended):

    ```bash
    # .env file
    DG_API_KEY=your_deepgram_key
    GROQ_API_KEY=your_groq_key
    GEMINI_API_KEY=your_gemini_key
    OPENROUTER_API_KEY=your_openrouter_key
    ```

## Dockerized Deployment 🐳

### Docker Compose Setup

The `docker-compose.yml` file defines two services:

- **`video-processor`**:  Builds and runs the Video Topic Splitter application.  It mounts the `./data` directory to `/app/data` inside the container for persistent storage of input and output.  It also sets environment variables for API keys.
- **`redis`**: Provides a Redis instance for caching analysis results.

**1. Configure Environment Variables:**

Create a `.env` file:

```
DEEPGRAM_API_KEY=YOUR_DEEPGRAM_API_KEY
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
OPENROUTER_API_KEY=YOUR_OPENROUTER_API_KEY
```

**2. Build and Run:**

```bash
docker-compose up --build
```

**3. Usage within Docker:**

```bash
docker exec -it video-processor video-topic-splitter -i /app/data/input.mp4 -o /app/data/output
```

### Dockerfile Explanation

The `Dockerfile` uses `linuxserver/ffmpeg` as a base image. Key steps:

- Installs system dependencies (Python, pip, build tools, Tesseract OCR).
- Creates a user `vts` (non-root) for security.
- Copies the application code.
- Installs Python dependencies.
- Sets the entrypoint to run the `video-topic-splitter` command, allowing for command-line arguments.

## Contributing 🧑‍💻

This project was generated as an exercise in utilizing Large Language Models (specifically Claude) to develop a Python application.

## License 📜

[MIT License](LICENSE)
