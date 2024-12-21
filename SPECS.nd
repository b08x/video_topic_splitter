# Technical Specifications

## Processing Pipeline

```mermaid
graph TD
    A[Input Video] --> B[Audio Processing]
    B --> C[Transcription]
    B --> D[Video Segmentation]
    C --> E[Topic Modeling]
    E --> F[Segment Analysis]
    D --> F
    F --> G[Final Output]

    subgraph "Audio Processing"
        B1[Normalize Audio] --> B2[Convert to Mono]
        B2 --> B3[Resample 16kHz]
        B3 --> B4[Apply Filters]
    end

    subgraph "AI Services"
        C1[Deepgram/Groq] --> C2[Speech to Text]
        F1[Gemini] --> F2[Visual Analysis]
    end
```

## Component Architecture

```mermaid
flowchart TB
    subgraph Core["Core Processing"]
        direction TB
        main[Main Pipeline] --> checkpoint[Checkpoint System]
        checkpoint --> processor[Process Manager]
    end

    subgraph Audio["Audio Processing"]
        direction TB
        ffmpeg[FFmpeg] --> normalize[Audio Normalization]
        normalize --> silence[Silence Detection]
        silence --> convert[Format Conversion]
    end

    subgraph AI["AI Services"]
        direction TB
        transcribe[Transcription APIs] --> topic[Topic Modeling]
        topic --> visual[Visual Analysis]
    end

    subgraph Storage["File Management"]
        direction TB
        project[Project Structure] --> cache[Cache System]
        cache --> results[Results Storage]
    end

    Core --> Audio
    Core --> AI
    Core --> Storage
```

## Detailed Specifications

### Audio Processing Parameters

#### FFmpeg Configuration
```bash
ffmpeg -i {input} \
    -af "highpass=f=200, \
         acompressor=threshold=-12dB:ratio=4:attack=5:release=50" \
    -ar 16000 \
    -ac 1 \
    -c:a aac \
    -b:a 128k \
    {output}
```

#### Audio Normalization
```bash
ffmpeg-normalize \
    -pr \
    -tp -9.0 \
    -nt rms \
    -prf "highpass=f=100" \
    -prf "dynaudnorm=p=0.4:s=15" \
    -pof "lowpass=f=8000" \
    -ar 48000 \
    -c:a pcm_s16le \
    --keep-loudness-range-target
```

#### Silence Detection
```python
SILENCE_PARAMS = {
    "duration": "1.5",    # Minimum silence duration in seconds
    "threshold": "-25"    # Silence threshold in dB
}
```

### AI Model Configurations

#### Deepgram Transcription
```python
DEEPGRAM_CONFIG = {
    "model": "nova-2",
    "language": "en",
    "features": {
        "topics": True,
        "intents": True,
        "smart_format": True,
        "punctuate": True,
        "paragraphs": True,
        "utterances": True,
        "diarize": True,
        "filler_words": True,
        "sentiment": True
    }
}
```

#### Groq Transcription
```python
GROQ_CONFIG = {
    "model": "whisper-large-v3",
    "temperature": 0.2,
    "response_format": "verbose_json",
    "language": "en"
}
```

#### Topic Modeling (LDA)
```python
LDA_PARAMS = {
    "num_topics": 5,
    "random_state": 100,
    "chunksize": 100,
    "passes": 10,
    "per_word_topics": True,
    "minimum_probability": 0.0
}
```

#### Gemini Visual Analysis
```python
GEMINI_CONFIG = {
    "model": "gemini-1.5-pro-latest",
    "analysis_prompt": """
        Analyze this video segment. 
        The transcript for this segment is: '{transcript}'. 
        Describe the main subject matter, key visual elements, 
        and how they relate to the transcript.
    """
}
```

## Recovery System

```mermaid
stateDiagram-v2
    [*] --> PROJECT_CREATED: Init
    PROJECT_CREATED --> AUDIO_PROCESSED: Process Audio
    AUDIO_PROCESSED --> TRANSCRIPTION_COMPLETE: Transcribe
    TRANSCRIPTION_COMPLETE --> TOPIC_MODELING_COMPLETE: Model Topics
    TOPIC_MODELING_COMPLETE --> SEGMENTS_IDENTIFIED: Identify Segments
    SEGMENTS_IDENTIFIED --> VIDEO_ANALYZED: Analyze Segments
    VIDEO_ANALYZED --> PROCESS_COMPLETE: Complete
    
    state "Error Recovery" as error {
        Failure --> LoadCheckpoint
        LoadCheckpoint --> ResumeProcess
        ResumeProcess --> ReturnToLastState
    }
```

## File Structure

```mermaid
graph TD
    A[Project Root] --> B[src/]
    A --> C[tests/]
    A --> D[docs/]
    
    B --> E[video_topic_splitter/]
    E --> F[__init__.py]
    E --> G[audio.py]
    E --> H[transcription.py]
    E --> I[topic_modeling.py]
    E --> J[video_analysis.py]
    E --> K[project.py]
    E --> L[core.py]
    E --> M[cli.py]
    
    C --> N[test_audio.py]
    C --> O[test_transcription.py]
    C --> P[test_topic_modeling.py]
```

## Output Format

### Segment Analysis JSON Structure
```json
{
  "segment_id": 1,
  "start_time": 0.0,
  "end_time": 120.5,
  "transcript": "...",
  "topic": {
    "id": 2,
    "keywords": ["..."],
    "confidence": 0.85
  },
  "visual_analysis": {
    "description": "...",
    "key_elements": ["..."],
    "transcript_correlation": 0.92
  }
}
```

## Project Organization

```mermaid
graph LR
    A[Input] --> B{Project Manager}
    B --> C[Audio Pipeline]
    B --> D[Transcription Service]
    B --> E[Topic Analysis]
    B --> F[Visual Processing]
    
    C --> G{Checkpoint System}
    D --> G
    E --> G
    F --> G
    
    G --> H[Results]
    G --> I[Recovery]
```

## Performance Considerations

### Resource Usage
- Audio Processing: ~2x input duration
- Transcription: API dependent
- Topic Modeling: O(n*k*i) where:
  - n = document length
  - k = number of topics
  - i = iteration count

### Recommended Specifications
- CPU: 4+ cores
- RAM: 8GB minimum
- Storage: 3x input video size
- GPU: Optional, improves video processing

## Attribution
All technical specifications above were provided by the user. The implementation of these specifications into working code was performed by Claude (Anthropic) AI. The documentation and diagrams were also generated by Claude based on the implementation details.