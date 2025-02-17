# Video Topic Splitter 🎬

**An AI-powered tool to automatically segment videos based on topic changes and analyze their content.**

## Table of Contents

- [Video Topic Splitter 🎬](#video-topic-splitter-)
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
  - [Project Structure 📂](#project-structure-)
  - [Configuration ⚙️](#configuration-️)
  - [Dockerized Deployment 🐳](#dockerized-deployment-)
    - [Docker Compose Setup](#docker-compose-setup)
    - [Dockerfile Explanation](#dockerfile-explanation)
  - [Contributing 🧑‍💻](#contributing-)
  - [License 📜](#license-)

## About 📖

This tool leverages advanced AI techniques, including audio processing, transcription, topic modeling, and visual analysis, to break down videos into meaningful segments based on shifts in conversation or subject matter. It then provides insights into each segment, including:

- **Dominant Topics:** Identifies the main subjects discussed in each segment.
- **Keywords:** Extracts relevant keywords to summarize the content of each segment.
- **Software Detection:** Optionally detects software applications and logos visible in the video.
- **Gemini Analysis:** Provides detailed insights using Google's Gemini model, summarizing and explaining the content of each segment with a focus on the chosen analysis register.
  - **IT Workflow:** Analyzes technical procedures, system commands, and software configurations.
  - **Generative AI:** Focuses on AI models, prompt engineering, and implementation details.
  - **Tech Support:** Identifies problem descriptions, diagnostic procedures, and resolution steps.

## Features ✨

- **Automatic Segmentation:** Intelligently splits videos into topic-coherent segments.
- **Topic Modeling:** Uses OpenRouter's phi-4 model for accurate topic identification.
- **Transcription:** Transcribes audio using Deepgram's speech recognition API.
- **Software Detection:** Detects the presence of software applications via OCR and logo recognition.
- **Gemini Analysis:** Leverages Google's Gemini for detailed segment analysis.
- **Checkpoint System:** Resumes processing from interruptions or errors.
- **YouTube Integration:** Downloads and processes videos directly from YouTube links.
- **Customizable Analysis:** Tailor the analysis with different registers (IT Workflow, Generative AI, Tech Support).
- **Screenshot Analysis:** Analyze individual screenshots for software applications and get Gemini insights.

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

## Tech Stack

- **Transcription**: Deepgram & Groq APIs for speech-to-text
- **Visual Analysis**: Google's Gemini for frame analysis
- **Topic Modeling**: OpenRouter's phi-4 model
- **Audio Processing**: FFmpeg for audio extraction and normalization
- **OCR**: Tesseract for text extraction from frames
- **Core**: Python with extensive use of async/await for performance

## Installation 💾

```bash
pip install video_topic_splitter
```

## Usage 💻

### Basic Video Processing

```bash
video-topic-splitter -i <input_video_path_or_youtube_url> -o <output_directory> --topics <number_of_topics> --register <register>
```

**Example:**

```bash
video-topic-splitter -i "https://www.youtube.com/watch?v=dQw4w9WgXcQ" -o output --topics 5 --register it-workflow 
```

### Transcribe Only

```bash
video-topic-splitter -i <input_video_path> -o <output_directory> --transcribe-only
```

### Analyze Screenshot

```bash
video-topic-splitter -i <image_path> -o <output_directory> --analyze-screenshot --screenshot-context "This is a screenshot of a user configuring a firewall." --software-list software_list.txt --logo-db logos/
```

### Advanced Options

- `--api`: Choose between `deepgram` (default) or `groq` for transcription.
- `--skip-unsilence`: Skip silence removal during audio preprocessing.
- `--software-list <path>`: Provide a text file with a list of software to detect (one per line).
- `--logo-db <path>`: Specify a directory containing logo template images for software detection.
  - Logo images should be named `<software_name>.png` (e.g., `firefox.png`, `vscode.png`).
- `--ocr-lang <language_code>`: Set the language for OCR (default: `eng`).
- `--logo-threshold <0.0-1.0>`: Adjust the confidence threshold for logo detection (default: 0.8).
- `--thumbnail-interval <seconds>`: Set the interval for generating thumbnails (default: 5).
- `--max-thumbnails <number>`: Limit the maximum number of thumbnails per segment (default: 5).
- `--min-thumbnail-confidence`: Minimum confidence for thumbnail-based software detection before analyzing more frames.

## Project Structure 📂

The tool creates a project folder for each video processed, containing:

- `audio/`: Extracted and processed audio files.
- `segments/`: Video segments generated based on topic changes.
- `thumbnails/`: Thumbnail images extracted from segments.
- `transcription.json`: Raw transcription data.
- `transcript.json`: Processed transcript with sentence segmentation.
- `results.json`: Final results including topic analysis, keywords, and segment metadata.
- `checkpoint.pkl`: Checkpoint file to resume processing.

## Configuration ⚙️

- **API Keys:** Set the following environment variables with your API keys:
  - `DG_API_KEY` (Deepgram)
  - `GROQ_API_KEY` (Groq)
  - `GEMINI_API_KEY` (Google Gemini)
  - `OPENROUTER_API_KEY` (OpenRouter)

## Dockerized Deployment 🐳

To simplify deployment and dependency management, the Video Topic Splitter can be run within a Docker container.  We provide a `Dockerfile` and `docker-compose.yml` to facilitate this process.

### Docker Compose Setup

The provided `docker-compose.yml` sets up two services:

- **`video-processor`**: This service builds the Video Topic Splitter image and runs the application. It mounts the `./data` directory to `/app/data` inside the container for persistent storage of project data.  It also defines environment variables for required API keys and depends on a Redis service for caching.
- **`redis`**: This service runs a Redis instance for caching analysis results, improving performance.  It exposes port 6389 on the host machine and uses a named volume `redis_data` for data persistence.

**1. Configure Environment Variables:**

Create a `.env` file in the root of the project and set the required API keys:

```bash
DEEPGRAM_API_KEY=YOUR_DEEPGRAM_API_KEY
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
OPENROUTER_API_KEY=YOUR_OPENROUTER_API_KEY
```

**2. Build and Run:**

```bash
docker-compose up --build
```

This command will build the `video-processor` image, pull the Redis image, and start both containers.  The application will be accessible inside the `video-processor` container.

**3. Usage within Docker:**

You can then run the `video-topic-splitter` command inside the running container:

```bash
docker exec -it video-processor video-topic-splitter -i /app/data/input.mp4 -o /app/data/output
```

Replace `/app/data/input.mp4` with the path to your input video file *inside* the container (mounted from your `./data` directory). Output will be saved to `/app/data/output` inside the container, which corresponds to `./data/output` on your host machine.

### Dockerfile Explanation

The `Dockerfile` uses the `linuxserver/ffmpeg` image as a base, providing pre-installed FFmpeg and related tools. Key steps include:

- Installing system dependencies (Python, pip, build tools, Tesseract OCR).
- Creating a dedicated user `vts` for security.
- Copying the application code into the container.
- Installing Python dependencies using `pip`.
- Setting the entrypoint to run the `video-topic-splitter` command.  The Dockerfile now uses a bash entrypoint to allow for dynamic command execution.

## Contributing 🧑‍💻

This is an exercise in using Large Language Models to craft an applicatio using Python.

## License 📜

[MIT License](LICENSE)
