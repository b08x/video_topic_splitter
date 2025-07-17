### 1. Project Initialization and Input Handling

* **Responsible Modules/Functions**: `cli.py:main()`, `project.py:create_project_folder()`, `utils.youtube.py:download_video()`
* **Explanation**: The application begins via the command-line interface, which parses user arguments. It then creates a unique, timestamped project directory to store all artifacts. The input is validated to be either a local video file or a YouTube URL. If it's a YouTube URL, the video is downloaded into the project folder before any other processing occurs.
* **Input**: Command-line arguments specifying the input video/URL and output directory.
* **Output**: A dedicated project folder, `source_video.mp4` (if downloaded from YouTube), and an initial `checkpoint.pkl` file marking the project's creation.

### 2. Audio Pre-processing

* **Responsible Modules/Functions**: `core.py:handle_audio_video()`, functions within `processing/audio/audio.py` (`normalize_audio`, `remove_silence`, `extract_audio`, `convert_to_mono_and_resample`).
* **Explanation**: The audio from the source video is extracted and optimized for accurate transcription. This involves several steps: normalizing the volume, removing or speeding up silent sections, extracting the processed audio into its own file, and finally converting it to a mono-channel, 16kHz audio file as required by the transcription service.
* **Input**: The source video file (e.g., `source_video.mp4`).
* **Output**: An audio-processed video (`unsilenced_video.mp4`), a separate clean audio file (`mono_resampled_audio.m4a`), and an updated `checkpoint.pkl` (Stage: `AUDIO_PROCESSED`).

### 3. Audio Transcription

* **Responsible Modules/Functions**: `core.py:handle_transcription()`, `api.deepgram.py:transcribe_file_deepgram()`
* **Explanation**: The processed audio file is sent to the Deepgram API for transcription. The API returns a detailed, timestamped transcript with speaker diarization and other metadata. The application saves both the raw API response and a simplified version containing just the utterance, start time, and end time.
* **Input**: The processed audio file (`mono_resampled_audio.m4a`).
* **Output**: `transcription.json` (the raw, detailed API response) and `transcript.json` (a simplified list of utterances). The checkpoint is updated to `TRANSCRIPTION_COMPLETE`.

### 4. Topic Modeling and Segmentation

* **Responsible Modules/Functions**: `analysis.topic_modeling.py:process_transcript()`, `TopicAnalyzer` class.
* **Explanation**: The `TopicAnalyzer` processes the `transcript.json`. It intelligently batches sentences and sends them to a generative AI model (via OpenRouter) with a specialized prompt. The AI analyzes the text to identify the main topic, associated keywords, and the relationship to the previous text block (e.g., `NEW`, `SHIFT`, `CONTINUATION`). Based on the AI's response, the transcript is segmented into distinct thematic chunks.
* **Input**: `transcript.json`.
* **Output**: An initial `results.json` file that contains the list of identified topics and the corresponding text segments. The checkpoint is updated to `TOPIC_MODELING_COMPLETE`.

### 5. Contextual Frame Extraction

* **Responsible Modules/Functions**: `analysis.visual_analysis.py:split_and_analyze_video()`, `analysis.frame_analysis.py:ContextualFrameAnalyzer`.
* **Explanation**: For each text segment identified in the previous step, the `ContextualFrameAnalyzer` programmatically extracts relevant video frames. It selects the best-quality frames from the start, end, and middle of the segment's duration, discarding blurry or low-contrast images. These selected frames are saved as images for individual analysis.
* **Input**: The audio-processed video (`unsilenced_video.mp4`) and the segment data from `results.json`.
* **Output**: Screenshot images (`.jpg` or `.png`) saved into subdirectories within the project folder (e.g., `/screenshots/segment_1/`).

### 6. Detailed Visual Content Analysis

* **Responsible Modules/Functions**: `analysis.frame_analysis.py:analyze_frame_with_context()`, `api.gemini.py:analyze_with_gemini()`, functions in `processing/ocr/` and `processing/software/`.
* **Explanation**: Each extracted frame undergoes a multi-layered analysis. First, Optical Character Recognition (`detect_software_names`) and template matching (`detect_software_logos`) are used to find pre-defined software names and logos. Then, the frame image—along with its context (the corresponding transcript text, topic, and any detected software)—is sent to the Gemini API for a rich, descriptive analysis of the visual content.
* **Input**: A single frame image and its associated context (transcript text, topic).
* **Output**: A detailed analysis for each frame, including detected software and a natural language description from Gemini.

### 7. Output Generation and Finalization

* **Responsible Modules/Functions**: `core.py:process_video()`, `analysis.visual_analysis.py`.
* **Explanation**: The application aggregates the results from the topic modeling (Step 4) and the detailed visual analyses (Step 6). A visual summary, including detected software and key visual elements, is generated for each segment. This final, comprehensive data structure is written to `results.json`, overwriting the preliminary version created in Step 4.
* **Input**: All intermediate topic and visual analysis data.
* **Output**: The final `results.json` file, containing each segment with its topic, keywords, transcript, and a detailed visual summary. The checkpoint is updated to `PROCESS_COMPLETE`.

### 8. Checkpointing (Ongoing Process)

* **Responsible Modules/Functions**: `project.py:save_checkpoint()`, `project.py:load_checkpoint()`.
* **Explanation**: After every major stage, a checkpoint is saved to a `checkpoint.pkl` file in the project directory. This file uses Python's `pickle` module to store the application's state, including the last completed stage and the file paths to any generated artifacts. If the process is interrupted, it can be resumed from the last checkpoint, preventing the need to re-run completed stages.
* **Input**: A stage identifier (e.g., `AUDIO_PROCESSED`) and a dictionary of relevant data (like file paths).
* **Output**: The `checkpoint.pkl` file in the project directory.
