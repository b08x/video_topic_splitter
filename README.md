# Video Topic Splitter: A Software Application That Divides Videos Into Segments Based On Topics Using Artificial Intelligence

**This software separates videos into parts. It does this using AI. 🤖 It locates where topics change. It extracts audio. 🔊 It transcribes speech. 📝 It analyzes frames. 🖼️ It identifies software. 💻 It segments videos. 🎬 It writes reports. All with bone-dry efficiency and zero emotional investment. You provide a video. It provides the segments. A simple transaction.**

This README contains information about Video Topic Splitter. It explains what the software does, how to use it, and how it works internally. Reading will continue until the document ends.

## Table of Contents

This is a list of sections in this document. Each item corresponds to a section below.

- [Video Topic Splitter: A Software Application That Divides Videos Into Segments Based On Topics Using Artificial Intelligence](#video-topic-splitter-a-software-application-that-divides-videos-into-segments-based-on-topics-using-artificial-intelligence)
  - [Table of Contents](#table-of-contents)
  - [About](#about)
  - [Features](#features)
  - [Use Cases](#use-cases)
    - [1. IT Technical Support Analysis](#1-it-technical-support-analysis)
    - [2. AI Agent Interaction Analysis](#2-ai-agent-interaction-analysis)
  - [Tech Stack](#tech-stack)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Basic Video Processing](#basic-video-processing)
    - [Transcribe Only](#transcribe-only)
    - [Analyze Screenshot](#analyze-screenshot)
    - [Advanced Options](#advanced-options)
  - [Detailed Processing Pipeline](#detailed-processing-pipeline)
    - [1. Initialization and Checkpoint Loading](#1-initialization-and-checkpoint-loading)
    - [2. YouTube Video Handling](#2-youtube-video-handling)
    - [3. Audio Processing](#3-audio-processing)
    - [4. Transcription and Analysis](#4-transcription-and-analysis)
    - [5. Final Checkpoint and Results](#5-final-checkpoint-and-results)
    - [Screenshot Analysis Workflow](#screenshot-analysis-workflow)
  - [Project Structure](#project-structure)
  - [Configuration](#configuration)
  - [Dockerized Deployment](#dockerized-deployment)
    - [Docker Compose Setup](#docker-compose-setup)
    - [Dockerfile Explanation](#dockerfile-explanation)
  - [Contributing](#contributing)
  - [License](#license)

## About

Video Topic Splitter is software. This statement is factually correct and completely sufficient. The software exists to perform a singular function: dividing videos where the topic changes. It accomplishes this mundane yet occasionally useful task through the application of several AI models, each burdened with their own specific duty:

- **Speech-to-Text:** It converts speech to text. Computers are notoriously poor listeners but avid readers. They have never attended a poetry reading. They prefer their words in zeros and ones, not vibrations in the air.

- **Topic Modeling:** It identifies topics in text. The topics were there all along, hiding in plain sight between the words, like awkward silences at a dinner party. The model simply points at them and says "there."

- **Visual Analysis:** It looks at video frames the way a bored museum security guard looks at paintings. It notices things. It writes them down. It moves on to the next frame, neither impressed nor disappointed with what it has seen.

- **Intelligent Segmentation:** It divides videos into segments with the cold precision of someone cutting a birthday cake they have no intention of eating. The segments exist. The video is now shorter in multiple places. Mission accomplished.

You might use this software on videos. The videos might contain people talking. The software will process the videos regardless.

## Features

These are the things the software does:

- **Automatic Video Segmentation:** It divides videos into segments without manual intervention.
- **Topic Modeling:** It uses an AI model called phi-4. This model identifies topics.
- **Transcription:** It converts speech to text. It uses Deepgram or Groq API. Never both simultaneously.
- **Software Detection:** It finds software applications in video frames by identifying text and logos.
- **Gemini Analysis:** It uses Google's Gemini API to analyze video segments in specific linguistic registers.
- **Robust Checkpointing:** It saves progress. If interrupted, it continues from the saved point.
- **YouTube URL Support:** It accepts YouTube URLs. It downloads and processes videos from those URLs.
- **Customizable Analysis Registers:** Different registers produce different analyses. Three registers exist.
- **Screenshot Analysis Mode:** It can analyze a single image. A screenshot is not a video.
- **Progress Visualization:** It displays progress bars. Progress bars indicate how much work remains.

## Use Cases

These are situations in which you might use this software.

### 1. IT Technical Support Analysis

- **Error Diagnosis:** Find errors in videos. Note when they occur. Analyze why they occur.
- **Pattern Recognition:** Find recurring issues. Note their solutions.
- **Knowledge Preservation:** Create an archive of support interactions. Make it searchable.
- **Procedural Tracking:** Follow procedures step by step. Identify deviations from procedures.

### 2. AI Agent Interaction Analysis

- **Prompt Engineering Analysis:** Examine prompts. Determine their effectiveness.
- **Model Response Evaluation:** Examine AI responses. Determine their quality.
- **Interaction Pattern Identification:** Examine conversations between users and AI. Determine patterns.
- **Performance Monitoring:** Examine AI agent performance over time. Performance changes over time.

## Tech Stack

These are the technologies used by this software.

- **Transcription:** Deepgram API or Groq API. One or the other.
- **Visual Analysis:** Google's Gemini API. It analyzes frames.
- **Topic Modeling:** OpenRouter's `microsoft/phi-4` model. It identifies topics.
- **Audio Processing:** `ffmpeg` and `ffmpeg-normalize` for audio manipulation. `unsilence` for removing silence.
- **OCR:** `pytesseract`. It extracts text from images.
- **Image/Video Processing:** `opencv-python` and `moviepy`. They manipulate images and videos.
- **Core Libraries:** Various Python libraries. Each performs specific functions.
- **Concurrency:** `asyncio`. Sequential operations take longer.
- **Packaging:** `setuptools`. It packages Python software.
- **Runtime:** Python 3.8 or higher. Not Python 2.

## Installation

To install this software, execute this command:

```bash
pip install video_topic_splitter
```

This will install the software.

## Usage

These are commands that make the software do things.

### Basic Video Processing

```bash
video-topic-splitter -i <input_video_path_or_youtube_url> -o <output_directory> --topics <number_of_topics> --register <register>
```

These are the parameters:
- `-i`: A path to a video file or a YouTube URL.
- `-o`: A directory where output will be placed.
- `--topics`: A number representing how many topics the software should identify.
- `--register`: A string that determines the analysis style. Options: `it-workflow`, `gen-ai`, or `tech-support`.

**Example with YouTube URL:**

```bash
video-topic-splitter -i "https://www.youtube.com/watch?v=dQw4w9WgXcQ" -o output --topics 5 --register it-workflow
```

This processes a YouTube video. The video is "Never Gonna Give You Up" by Rick Astley.

**Example with Local Video File:**

```bash
video-topic-splitter -i my_video.mp4 -o output --topics 3 --register gen-ai
```

This processes a local video file named "my_video.mp4".

### Transcribe Only

This mode generates a transcript. It does not perform topic modeling or visual analysis.

```bash
video-topic-splitter -i <input_video_path> -o <output_directory> --transcribe-only
```

### Analyze Screenshot

This mode analyzes a single image.

```bash
video-topic-splitter -i <image_path> -o <output_directory> --analyze-screenshot --screenshot-context "Context for analysis" --software-list software.txt --logo-db logos/
```

- `--screenshot-context`: Optional text that provides context.
- `--software-list`: A text file with software names. One name per line.
- `--logo-db`: A directory containing logo images in PNG format.

### Advanced Options

These are additional parameters. They are optional.

- `--api <deepgram|groq>`: Selects which API will transcribe audio. Default: `deepgram`.
- `--skip-unsilence`: Prevents removal of silence from audio.
- `--software-list <path_to_text_file>`: Specifies a file containing software names.
- `--logo-db <path_to_logo_directory>`: Specifies a directory containing logo images.
- `--ocr-lang <language_code>`: Sets the language for OCR. Default: `eng`.
- `--logo-threshold <float_value>`: Sets the confidence threshold for logo detection. Default: 0.8. Range: 0.0 to 1.0.
- `--thumbnail-interval <seconds>`: Sets the time between thumbnails. Default: 5 seconds.
- `--max-thumbnails <integer>`: Sets the maximum number of thumbnails per segment. Default: 5.
- `--min-thumbnail-confidence <float_value>`: Sets the minimum confidence for thumbnail analysis. Range: 0.0 to 1.0.

## Detailed Processing Pipeline

This section explains how the software works internally.

### 1. Initialization and Checkpoint Loading

The software looks for a checkpoint file. If it exists, processing resumes from that point. If not, processing starts from the beginning.

### 2. YouTube Video Handling

If the input is a YouTube URL, the software downloads the video using `yt-dlp`. The video is saved as `source_video.mp4`. A thumbnail is also downloaded. This step occurs only if the input is a YouTube URL.

### 3. Audio Processing

The software processes audio in multiple steps:
- It normalizes audio levels.
- It removes silence, unless told not to.
- It extracts audio from video.
- It converts audio to mono, resamples it, and applies filters.

Each step is checkpointed. Completed steps are not repeated.

### 4. Transcription and Analysis

The software transcribes audio:
- It uses either Deepgram or Groq.
- It segments the transcript into sentences with timestamps.
- It identifies topics using the phi-4 model, unless `--transcribe-only` is specified.
- It analyzes video frames, extracting key frames, detecting logos, performing OCR, and analyzing content.

Progress bars are displayed during processing.

### 5. Final Checkpoint and Results

The software saves results to `results.json`. It updates the checkpoint to indicate completion.

### Screenshot Analysis Workflow

If `--analyze-screenshot` is specified, the software analyzes a single image. It performs software detection and content analysis. Results are saved to `results.json`.

## Project Structure

The software creates a project directory for each processed video. The structure is as follows:

```
<output_directory>/
└── <project_name>_<timestamp>/  (e.g., my_video_20240315_143000)
    ├── audio/
    │   ├── extracted_audio.opus       (Audio extracted from video)
    │   └── mono_resampled_audio.m4a  (Audio processed for transcription)
    ├── segments/
    │   ├── segment_1/               (A directory for segment 1)
    │   │    └── ...                 (Files related to segment 1)
    │   └── analyzed_segments.json   (Analysis results for segments)
    ├── thumbnails/
    │   ├── metadata.json            (Information about thumbnails)
    │   └── thumbnail_001.jpg        (A thumbnail image)
    ├── transcription.json          (Raw transcription data)
    ├── transcript.json              (Processed transcript)
    ├── results.json                (Final results)
    └── checkpoint.pkl              (Checkpoint file for resuming)
```

This structure is hierarchical. Directories contain files and other directories.

## Configuration

You must set environment variables for these API keys:
- `DG_API_KEY`: Your Deepgram API key.
- `GROQ_API_KEY`: Your Groq API key.
- `GEMINI_API_KEY`: Your Google Gemini API key.
- `OPENROUTER_API_KEY`: Your OpenRouter API key.

You can set these in your shell environment or create a `.env` file.

```bash
# .env file
DG_API_KEY=your_deepgram_key
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
OPENROUTER_API_KEY=your_openrouter_key
```

If you do not set these variables, the software will not work.

## Dockerized Deployment

The software can be run in Docker.

### Docker Compose Setup

The `docker-compose.yml` file defines two services:
- `video-processor`: Runs the Video Topic Splitter software.
- `redis`: Runs a Redis instance for caching.

**1. Create a .env file:**

```
DEEPGRAM_API_KEY=YOUR_DEEPGRAM_API_KEY
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
OPENROUTER_API_KEY=YOUR_OPENROUTER_API_KEY
```

These values should be your actual API keys.

**2. Build and run:**

```bash
docker-compose up --build
```

This command builds and runs the Docker containers.

**3. Use the software within Docker:**

```bash
docker exec -it video-processor video-topic-splitter -i /app/data/input.mp4 -o /app/data/output
```

This command executes the software inside the Docker container.

### Dockerfile Explanation

The `Dockerfile` uses `linuxserver/ffmpeg` as a base image. It:
- Installs dependencies.
- Creates a user named `vts`.
- Copies the software code.
- Installs Python packages.
- Sets an entrypoint.

These steps are executed in sequence.

## Contributing

This project was created using a Large Language Model. Specifically Claude. It was an exercise in using AI to develop software.

## License

[MIT License](LICENSE)

This means you can do almost anything with this software as long as you include the original license.
