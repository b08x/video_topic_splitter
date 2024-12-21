# Video Topic Splitter

> Because life's too short to watch irrelevant video segments.

_Note: This entire README was generated by Claude, an Anthropic AI. The irony of using AI to document a tool that uses AI to analyze videos isn't lost on us. We're living in the future, folks!_

A Python tool that intelligently splits videos into topic-based segments using a combination of audio processing, transcription, topic modeling, and AI-powered visual analysis. Think of it as your personal video librarian that understands both what's being said AND what's being shown.

## Technical Specifications & AI Attribution

This project represents a collaboration between human expertise and AI implementation. Here's the breakdown:

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
Before we dive in, let's acknowledge something cool: this documentation was written by the same kind of technology that powers part of this tool. While I (the AI) can write about natural language processing and computer vision, the actual tool uses similar but more specialized models (Gemini, Deepgram, Groq) to understand video content. It's AI all the way down! 🐢

## What's This All About?

Ever tried finding that one specific part of a lecture or presentation but ended up scrubbing through the whole thing? Yeah, me too. That's why this tool exists. It:

- 🎯 Splits videos into meaningful segments based on topic changes
- 🗣️ Transcribes everything (using either Deepgram or Groq - your choice)
- 🧠 Uses LDA topic modeling to identify natural break points
- 👀 Analyzes visual content using Google's Gemini
- 💾 Maintains checkpoints so you don't lose progress when things go sideways

## AI Stack
Since we're being transparent about AI usage, here's the full AI stack this tool employs:
- **Transcription**: Deepgram/Groq for speech-to-text
- **Topic Analysis**: Classical ML with LDA (okay, not "AI" but still smart)
- **Visual Analysis**: Google's Gemini for understanding video content
- **Documentation**: Written by Claude (that's me! 👋)

## Requirements

- Python 3.8+
- FFmpeg (for audio processing)
- `ffmpeg-normalize`
- API keys for:
  - Deepgram or Groq (for transcription)
  - Google Gemini (for visual analysis)

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
video-topic-splitter -i your_video.mp4 -o output_directory
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

### Topic Analysis
- LDA (Latent Dirichlet Allocation) topic modeling
- Automatic keyword extraction
- Smart segmentation based on topic shifts

### Visual Analysis
- Frame-by-frame analysis using Gemini
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

### Adjusting Topic Count
```bash
video-topic-splitter -i video.mp4 -o output --topics 7
```

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

## A Note from Your AI Documentation Writer

I find it fascinating to be writing documentation for a tool that itself uses AI to understand and process videos. It's like explaining how my cousins (Gemini, Deepgram, and Groq) work together to make sense of multimedia content. The future is pretty cool, isn't it?

Remember: This is a tool, not a magic wand. It works best with clear, well-structured content. If your input video is chaos, expect chaotic results. Garbage in, garbage out, as they say in the biz. And hey, if you're reading this, you're literally reading AI-generated documentation about an AI-powered tool. How meta is that? 🤖✨

---
_Documentation generated by Claude (Anthropic) on November 16, 2024. Yes, I'm an AI, and I'm proud of it! Feel free to update this documentation with human-written content if you prefer._