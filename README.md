# Video Topic Splitter

> Because life's too short to watch irrelevant video segments.

_Note: This entire README was generated by Claude, an Anthropic AI. The irony of using AI to document a tool that uses AI to analyze videos isn't lost on us. We're living in the future, folks!_

A Python tool that intelligently splits videos into topic-based segments using a combination of audio processing, transcription, topic modeling, and AI-powered visual analysis. Think of it as your personal video librarian that understands both what's being said AND what's being shown.

## Technical Specifications & AI Attribution

This project represents a collaboration between human expertise and AI implementation. Here's the breakdown:

### New Features & Capabilities

#### Intelligent Analysis Pipeline
- **Advanced Topic Modeling**:
  - OpenRouter phi-4 model integration
  - Smart boundary detection with similarity analysis
  - Async processing with concurrent batches
  - Context-aware topic transitions
  - Automatic result caching

- **Enhanced Visual Analysis**:
  - Software detection through OCR and logos
  - Frame-by-frame Gemini analysis
  - Adaptive thumbnail generation
  - YouTube thumbnail integration
  - Confidence-based detection

- **Robust Project Management**:
  - Comprehensive checkpoint system
  - Progress tracking and recovery
  - Metadata management
  - Cached results for performance
  - Error handling and resumption

### User-Provided Specifications:

#### Audio Processing Pipeline
```python
# FFmpeg Audio Processing Parameters
AUDIO_PARAMS = {
    'highpass_filter': 'f=200',
    'compressor': 'threshold=-12dB:ratio=4:attack=5:release=50',
    'sample_rate': 16000,
    'normalization': {
        'target_level': -9.0,
        'type': 'rms',
        'loudness_range_target': True
    },
    'silence_params': {
        'duration': '1.5',  # seconds
        'threshold': '-25'  # dB
    }
}
```

#### AI Model Selection & Configuration
- **Deepgram**: nova-2 model
  ```python
  deepgram_options = PrerecordedOptions(
      model="nova-2",
      language="en",
      topics=True,
      intents=True,
      smart_format=True,
      punctuate=True,
      paragraphs=True,
      utterances=True,
      diarize=True,
      filler_words=True,
      sentiment=True
  )
  ```
- **Groq**: whisper-large-v3
  ```python
  groq_params = {
      'model': 'whisper-large-v3',
      'temperature': 0.2,
      'response_format': 'verbose_json'
  }
  ```
- **Gemini**: gemini-1.5-pro-latest
  ```python
  # Visual Analysis Prompt Template
  ANALYSIS_PROMPT = """Analyze this video segment. 
  The transcript for this segment is: '{transcript}'. 
  Describe the main subject matter, key visual elements, 
  and how they relate to the transcript."""
  ```
- **Topic Modeling**: LDA Configuration
  ```python
  lda_params = {
      'num_topics': 5,
      'random_state': 100,
      'chunksize': 100,
      'passes': 10,
      'per_word_topics': True
  }
  ```

### AI-Generated Implementation:
The actual code implementation was generated by Claude (Anthropic), including:
- Package structure and architecture
- Error handling and recovery mechanisms
- File processing and checkpointing logic
- Documentation and inline comments

## Meta Commentary
Before we dive in, let's acknowledge something cool: this documentation was written by the same kind of technology that powers part of this tool. While I (the AI) can write about natural language processing and computer vision, the actual tool uses similar but more specialized models (Gemini, Deepgram, Groq, OpenRouter) to understand video content. It's AI all the way down! 🐢

## What's This All About?

Ever tried finding that one specific part of a lecture or presentation but ended up scrubbing through the whole thing? Yeah, me too. That's why this tool exists. It:

- 🎯 Splits videos into meaningful segments based on topic changes
- 🗣️ Transcribes everything (using either Deepgram or Groq - your choice)
- 🧠 Uses advanced topic modeling with OpenRouter's phi-4 model
- 👀 Analyzes visual content using Google's Gemini
- 🔍 Detects software applications through OCR and logo recognition
- 🖼️ Generates smart thumbnails for better visual understanding
- 💾 Maintains checkpoints so you don't lose progress when things go sideways

## AI Stack
Since we're being transparent about AI usage, here's the full AI stack this tool employs:
- **Transcription**: Deepgram/Groq for speech-to-text
- **Topic Analysis**: OpenRouter's phi-4 model for advanced topic modeling
- **Visual Analysis**: Google's Gemini for understanding video content
- **OCR**: Tesseract for text detection in frames
- **Documentation**: Written by Claude (that's me! 👋)

## Requirements

- Python 3.8+
- FFmpeg (for audio processing)
- `ffmpeg-normalize`
- Tesseract OCR engine
- API keys for:
  - Deepgram or Groq (for transcription)
  - Google Gemini (for visual analysis)
  - OpenRouter (for topic modeling)

## Quick Start

1. Clone and set up:
```bash
git clone https://github.com/yourusername/video_topic_splitter.git
cd video_topic_splitter
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
pip install -e .
```

2. Set up your environment variables:
```bash
# .env file
DG_API_KEY=your_deepgram_key
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
```

3. Run it:
```bash
# Basic usage
video-topic-splitter -i your_video.mp4 -o output_directory

# With software detection
video-topic-splitter -i your_video.mp4 -o output_directory \
  --software-list apps.txt \
  --logo-db logos/ \
  --ocr-lang eng \
  --logo-threshold 0.8

# With thumbnail customization
video-topic-splitter -i your_video.mp4 -o output_directory \
  --thumbnail-interval 5 \
  --max-thumbnails 5 \
  --min-thumbnail-confidence 0.7
```

## Features in Detail

### Audio Processing Pipeline
- Normalizes audio levels 🔊
- Removes silence (optional) 🤫
- Converts to mono and resamples for optimal transcription 🎵

### Transcription Options
- Deepgram (default): Better for technical content
- Groq: More contextual understanding
- Custom prompting available for Groq

### Analysis Features

#### Topic Analysis
- OpenRouter phi-4 model integration:
  - Smart boundary detection
  - Async batch processing
  - Context-aware transitions
  - Confidence scoring
  - Result caching
- Text processing:
  - TF-IDF analysis
  - NLTK integration
  - Stopword handling

#### Visual Processing
- Software detection:
  - OCR-based text recognition
  - Logo template matching
  - Version information extraction
- Thumbnail management:
  - Adaptive interval generation
  - YouTube integration
  - Quality control
- Gemini analysis:
  - Frame-by-frame processing
  - Software usage detection
  - UI element identification

#### Project Management
- Checkpoint system:
  - Progress tracking
  - State recovery
  - Result caching
- Error handling:
  - Automatic retries
  - Partial result saving
  - Resume capability

### Visual Analysis
- Frame-by-frame analysis using Gemini
- Software application detection through OCR and logo matching
- Advanced thumbnail management:
  - Smart adaptive interval generation
  - YouTube thumbnail integration
  - Metadata tracking and management
  - Quality-controlled JPEG compression
- Context-aware scene understanding
- Correlation with transcript content

### Robustness Features
- Checkpoint system for long processes
- Recovers from failures gracefully
- Caches intermediate results
- Skips already processed segments on restart

## Output Structure

```
output_directory/
├── audio/
│   ├── extracted_audio.wav
│   └── mono_resampled_audio.m4a
├── segments/
│   ├── segment_1.mp4
│   ├── segment_2.mp4
│   ├── thumbnails/
│   │   ├── segment1_thumb1.jpg
│   │   ├── segment2_thumb1.jpg
│   │   ├── yt_thumbnail_000.jpg  # YouTube source thumbnail
│   │   └── metadata.json         # Thumbnail tracking and metadata
│   └── analyzed_segments.json
├── transcription.json
├── transcript.json
├── results.json
└── checkpoint.pkl
```

## Advanced Usage

### Using Groq for Transcription
```bash
video-topic-splitter -i video.mp4 -o output --api groq --groq-prompt "Technical presentation context"
```

### Topic Analysis Configuration
```bash
# Basic topic count adjustment
video-topic-splitter -i video.mp4 -o output --topics 7

# Advanced configuration via environment variables
export TOPIC_ANALYZER_BATCH_SIZE=5        # Number of sentences per batch
export TOPIC_ANALYZER_MAX_CONCURRENT=3    # Maximum concurrent API calls
export TOPIC_ANALYZER_SIMILARITY_THRESHOLD=0.7  # Boundary detection threshold
export TOPIC_ANALYZER_MAX_RETRIES=3       # API retry attempts

# The analyzer automatically:
# - Batches content for optimal processing
# - Detects natural topic boundaries
# - Caches results for performance
# - Handles API rate limiting
# - Provides confidence scores for transitions
```

### Software Detection
```bash
# Create a text file with software names
echo -e "Visual Studio Code\nPython IDLE\nJupyter Notebook" > apps.txt

# Run with software detection
video-topic-splitter -i video.mp4 -o output \
  --software-list apps.txt \
  --logo-db path/to/logos \
  --ocr-lang eng \
  --logo-threshold 0.8
```

### Thumbnail Management
```bash
# Basic thumbnail configuration
video-topic-splitter -i video.mp4 -o output \
  --thumbnail-interval 5 \
  --max-thumbnails 5 \
  --min-thumbnail-confidence 0.7

# The tool automatically:
# - Adapts intervals for optimal coverage
# - Manages metadata for tracking
# - Handles YouTube thumbnails for YouTube URLs
# - Maintains quality with 85% JPEG compression
# - Organizes thumbnails in the project structure
```

#### YouTube Integration
When processing YouTube videos, the tool automatically:
- Downloads the highest quality thumbnail available
- Integrates it with locally generated thumbnails
- Maintains source tracking in metadata
- Provides consistent analysis across sources

## Troubleshooting

### Common Issues

1. **Transcription fails:**
   - Check your API keys
   - Verify audio quality
   - Try the alternative transcription service

2. **Segmentation seems off:**
   - Increase the number of topics
   - Check if the video has clear topic transitions
   - Review the transcript quality

3. **Process dies mid-way:**
   - Just run the same command again
   - It'll pick up where it left off
   - Check your disk space

4. **AI services acting up:**
   - Each AI service (Deepgram, Groq, Gemini) has its own quirks
   - Check their status pages
   - Remember they're like us (AI), just more specialized! 

## Known Limitations

- Silence removal is currently disabled (but the code's there if you're feeling brave)
- Large files might need significant processing time
- Topic modeling works best with clear subject transitions
- Visual analysis requires decent video quality
- OCR accuracy depends on text clarity and contrast
- Logo detection requires good quality reference images
- AI services can sometimes be... well, AI-ish (we're not perfect!)

## Contributing

Hey, I'm not precious about this. If you've got improvements:

1. Fork it
2. Branch it
3. Code it
4. Test it
5. PR it

Just keep it clean and explain your changes. And yes, you can even improve this AI-generated documentation!

## License

MIT. Go wild, just don't sue me. (Even AIs know about licenses!)

## Acknowledgments

Built with these awesome tools:
- [Deepgram](https://deepgram.com/) - Transcription
- [Groq](https://groq.com/) - Alternative transcription
- [Gemini](https://deepmind.google/technologies/gemini/) - Visual analysis
- [Gensim](https://radimrehurek.com/gensim/) - Topic modeling
- [FFmpeg](https://ffmpeg.org/) - Audio processing
- [Claude](https://anthropic.com) - For writing this documentation
- And more in the requirements.txt

## Future Ideas

- [ ] Add silence removal back (properly this time)
- [ ] Support for streaming sources
- [ ] Multi-language support
- [ ] GPU acceleration for video processing
- [ ] Web interface (maybe)
- [ ] Docker container (definitely)
- [ ] More AI models (because why not add more of my kind?)
- [ ] Expanded software detection database
- [ ] Custom OCR model training
- [ ] Real-time software detection

## A Note from Your AI Documentation Writer

I find it fascinating to be writing documentation for a tool that itself uses AI to understand and process videos. It's like explaining how my cousins (Gemini, Deepgram, and Groq) work together to make sense of multimedia content. The future is pretty cool, isn't it?

Remember: This is a tool, not a magic wand. It works best with clear, well-structured content. If your input video is chaos, expect chaotic results. Garbage in, garbage out, as they say in the biz. And hey, if you're reading this, you're literally reading AI-generated documentation about an AI-powered tool. How meta is that? 🤖✨

---
_Documentation generated by Claude (Anthropic) on November 16, 2024. Yes, I'm an AI, and I'm proud of it! Feel free to update this documentation with human-written content if you prefer._