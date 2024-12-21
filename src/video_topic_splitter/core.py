#!/usr/bin/env python3
"""Core processing functionality for video topic splitter."""

import os
import json
from dotenv import load_dotenv
import videogrep
from deepgram import DeepgramClient, PrerecordedOptions
from groq import Groq

from .constants import CHECKPOINTS
from .project import save_checkpoint
from .audio import (
    normalize_audio,
    remove_silence,
    extract_audio,
    convert_to_mono_and_resample
)
from .transcription import (
    transcribe_file_deepgram,
    transcribe_file_groq,
    save_transcription,
    save_transcript
)
from .topic_modeling import (
    preprocess_text,
    perform_topic_modeling,
    identify_segments,
    generate_metadata
)
from .video_analysis import split_and_analyze_video

load_dotenv()

def handle_audio_video(video_path, project_path):
    """Process audio from video file with checkpointing and file existence checks."""
    audio_dir = os.path.join(project_path, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    # Define paths
    normalized_video_path = os.path.join(project_path, "normalized_video.mkv")
    unsilenced_video_path = os.path.join(project_path, "unsilenced_video.mkv")
    raw_audio_path = os.path.join(audio_dir, "extracted_audio.wav")
    mono_resampled_audio_path = os.path.join(audio_dir, "mono_resampled_audio.m4a")

    # Check for existing processed files
    if os.path.exists(unsilenced_video_path) and os.path.exists(mono_resampled_audio_path):
        print("Found existing processed audio files. Using cached versions.")
        return unsilenced_video_path, mono_resampled_audio_path

    # Normalize audio if needed
    if not os.path.exists(normalized_video_path):
        print("Normalizing audio...")
        normalize_result = normalize_audio(video_path, normalized_video_path)
        if normalize_result["status"] == "error":
            print(f"Error during audio normalization: {normalize_result['message']}")
            raise RuntimeError("Audio normalization failed")
        else:
            print(normalize_result["message"])
    else:
        print("Using existing normalized video file.")

    # For now, we'll just copy the normalized video to unsilenced path since silence removal is disabled
    if not os.path.exists(unsilenced_video_path):
        # import shutil
        # shutil.copy2(normalized_video_path, unsilenced_video_path)
        # print("Created unsilenced video file (silence removal currently disabled).")
        silence_removal_result = remove_silence(
            normalized_video_path, unsilenced_video_path
        )
        if silence_removal_result["status"] == "error":
            print(f"Error during silence removal: {silence_removal_result['message']}")
            # Handle the error (e.g., exit or continue without silence removal)
        else:
            print(silence_removal_result["message"])
    # Extract audio if needed
    if not os.path.exists(raw_audio_path):
        print("Extracting audio from video...")
        try:
            extract_audio(unsilenced_video_path, raw_audio_path)
            print("Audio extraction complete.")
        except Exception as e:
            print(f"Error during audio extraction: {str(e)}")
            raise

    # Convert to mono and resample if needed
    if not os.path.exists(mono_resampled_audio_path):
        print("Converting audio to mono and resampling...")
        conversion_result = convert_to_mono_and_resample(raw_audio_path, mono_resampled_audio_path)
        if conversion_result["status"] == "error":
            print(f"Error during audio conversion: {conversion_result['message']}")
            raise RuntimeError("Audio conversion failed")
        else:
            print(conversion_result["message"])
    else:
        print("Using existing mono resampled audio file.")

    # Save checkpoint only if we've successfully processed everything
    if os.path.exists(unsilenced_video_path) and os.path.exists(mono_resampled_audio_path):
        save_checkpoint(project_path, CHECKPOINTS['AUDIO_PROCESSED'], {
            'unsilenced_video_path': unsilenced_video_path,
            'mono_resampled_audio_path': mono_resampled_audio_path
        })
        print("Audio processing checkpoint saved.")

    return unsilenced_video_path, mono_resampled_audio_path

def process_transcript(transcript, project_path, num_topics=5):
    """Process transcript for topic modeling and segmentation."""
    full_text = " ".join([sentence["content"] for sentence in transcript])
    preprocessed_subjects = preprocess_text(full_text)
    lda_model, corpus, dictionary = perform_topic_modeling(preprocessed_subjects, num_topics)
    
    save_checkpoint(project_path, CHECKPOINTS['TOPIC_MODELING_COMPLETE'], {
        'lda_model': lda_model,
        'corpus': corpus,
        'dictionary': dictionary
    })

    segments = identify_segments(transcript, lda_model, dictionary, num_topics)
    save_checkpoint(project_path, CHECKPOINTS['SEGMENTS_IDENTIFIED'], {
        'segments': segments
    })

    metadata = generate_metadata(segments, lda_model)

    results = {
        "topics": [
            {
                "topic_id": topic_id,
                "words": [word for word, _ in lda_model.show_topic(topic_id, topn=10)],
            }
            for topic_id in range(num_topics)
        ],
        "segments": metadata,
    }

    results_path = os.path.join(project_path, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

    return results

def handle_transcription(video_path, audio_path, project_path, api="deepgram", num_topics=2, groq_prompt=None):
    """Handle transcription of video/audio content."""
    segments_dir = os.path.join(project_path, "segments")
    os.makedirs(segments_dir, exist_ok=True)

    print("Parsing transcript with Videogrep...")
    transcript = videogrep.parse_transcript(video_path)
    print("Transcript parsing complete.")

    if not transcript:
        print("No transcript found. Transcribing audio...")
        deepgram_key = os.getenv("DG_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")

        if not deepgram_key:
            raise ValueError("DG_API_KEY environment variable is not set")
        if not groq_key and api == "groq":
            raise ValueError("GROQ_API_KEY environment variable is not set")

        if api == "deepgram":
            deepgram_client = DeepgramClient(deepgram_key)
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
                sentiment=True,
            )
            transcription = transcribe_file_deepgram(deepgram_client, audio_path, deepgram_options)
            transcript = [
                {
                    "content": utterance["transcript"],
                    "start": utterance["start"],
                    "end": utterance["end"],
                }
                for utterance in transcription["results"]["utterances"]
            ]
        else:  # Groq
            groq_client = Groq(api_key=groq_key)
            transcription = transcribe_file_groq(groq_client, audio_path, prompt=groq_prompt)
            transcript = [
                {
                    "content": segment["text"],
                    "start": segment["start"],
                    "end": segment["end"],
                }
                for segment in transcription["segments"]
            ]

        save_transcription(transcription, project_path)
        save_transcript(transcript, project_path)
        
    save_checkpoint(project_path, CHECKPOINTS['TRANSCRIPTION_COMPLETE'], {
        'transcript': transcript
    })

    results = process_transcript(transcript, project_path, num_topics)

    # Split the video and analyze segments
    try:
        analyzed_segments = split_and_analyze_video(video_path, results["segments"], segments_dir)
        
        # Update results with analyzed segments
        results["analyzed_segments"] = analyzed_segments
        
        # Save results after successful analysis
        results_path = os.path.join(project_path, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
            
        print(f"Successfully analyzed {len(analyzed_segments)} segments")
    except Exception as e:
        print(f"Error during video analysis: {str(e)}")
        # Load any segments that were successfully analyzed
        try:
            analyzed_segments = split_and_analyze_video(video_path, [], segments_dir)  # Just load existing
            results["analyzed_segments"] = analyzed_segments
            print(f"Recovered {len(analyzed_segments)} previously analyzed segments")
        except Exception as load_error:
            print(f"Could not load analyzed segments: {str(load_error)}")
            results["analyzed_segments"] = []

    # Save updated results
    results_path = os.path.join(project_path, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    save_checkpoint(project_path, CHECKPOINTS['VIDEO_ANALYZED'], {
        'results': results
    })

    return results

def process_video(video_path, project_path, api="deepgram", num_topics=2, groq_prompt=None):
    """Main video processing pipeline."""
    from .project import load_checkpoint
    
    checkpoint = load_checkpoint(project_path)
    
    if checkpoint is None or checkpoint['stage'] < CHECKPOINTS['AUDIO_PROCESSED']:
        unsilenced_video_path, mono_resampled_audio_path = handle_audio_video(
            video_path, project_path
        )
    else:
        unsilenced_video_path = checkpoint['data']['unsilenced_video_path']
        mono_resampled_audio_path = checkpoint['data']['mono_resampled_audio_path']

    if checkpoint is None or checkpoint['stage'] < CHECKPOINTS['VIDEO_ANALYZED']:
        results = handle_transcription(
            unsilenced_video_path,
            mono_resampled_audio_path,
            project_path,
            api,
            num_topics,
            groq_prompt
        )
    else:
        results = checkpoint['data']['results']

    save_checkpoint(project_path, CHECKPOINTS['PROCESS_COMPLETE'], {
        'results': results
    })

    return results