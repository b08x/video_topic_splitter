#!/usr/bin/env python3
"""Video analysis and segmentation functionality."""


import os
import json
import progressbar
from moviepy.editor import VideoFileClip
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def load_analyzed_segments(segments_dir):
    """Load any previously analyzed segments."""
    analysis_file = os.path.join(segments_dir, "analyzed_segments.json")
    if os.path.exists(analysis_file):
        with open(analysis_file, 'r') as f:
            return json.load(f)
    return []

def save_analyzed_segments(segments_dir, analyzed_segments):
    """Save the current state of analyzed segments."""
    analysis_file = os.path.join(segments_dir, "analyzed_segments.json")
    with open(analysis_file, 'w') as f:
        json.dump(analyzed_segments, f, indent=2)

def analyze_segment_with_gemini(segment_path, transcript):
    """Analyze a video segment using Google's Gemini model."""
    print(f"Analyzing segment: {segment_path}")
    
    # Check if Gemini API key is configured
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY environment variable is not set")
        
    try:
        # Load the video segment as an image (first frame)
        video = VideoFileClip(segment_path)
        frame = video.get_frame(0)
        image = Image.fromarray(frame)
        video.close()

        # Prepare the prompt
        prompt = f"Analyze this video segment. The transcript for this segment is: '{transcript}'. Describe the main subject matter, key visual elements, and how they relate to the transcript."

        # Initialize Gemini model
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        
        # Generate content
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        print(f"Error analyzing segment with Gemini: {str(e)}")
        return f"Analysis failed: {str(e)}"

def split_and_analyze_video(input_video, segments, output_dir):
    """Split video into segments and analyze each segment with checkpoint support."""
    print("Splitting video into segments and analyzing...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load any previously analyzed segments
    analyzed_segments = load_analyzed_segments(output_dir)
    
    # Create a map of existing analyses by segment_id
    existing_analyses = {seg["segment_id"]: seg for seg in analyzed_segments}
    
    try:
        video = VideoFileClip(input_video)
        total_segments = len(segments)
        
        print(f"Processing {total_segments} segments...")
        for i, segment in enumerate(progressbar.progressbar(segments)):
            segment_id = i + 1
            output_path = os.path.join(output_dir, f"segment_{segment_id}.mp4")
            
            # Skip if this segment has already been fully processed
            if segment_id in existing_analyses:
                print(f"\nSkipping segment {segment_id} (already processed)")
                continue
                
            try:
                # Extract and save the segment
                if not os.path.exists(output_path):
                    start_time = segment["start_time"]
                    end_time = segment["end_time"]
                    segment_clip = video.subclip(start_time, end_time)
                    segment_clip.write_videofile(
                        output_path,
                        codec="libx264",
                        audio_codec="aac",
                        verbose=False,
                        logger=None
                    )
                
                # Analyze the segment
                gemini_analysis = analyze_segment_with_gemini(
                    output_path,
                    segment["transcript"]
                )
                
                # Create the analysis result
                analysis_result = {
                    "segment_id": segment_id,
                    "start_time": segment["start_time"],
                    "end_time": segment["end_time"],
                    "transcript": segment["transcript"],
                    "topic": segment["dominant_topic"],
                    "keywords": segment["top_keywords"],
                    "gemini_analysis": gemini_analysis,
                }
                
                # Add to our results and save immediately
                analyzed_segments.append(analysis_result)
                save_analyzed_segments(output_dir, analyzed_segments)
                
                print(f"\nCompleted segment {segment_id}/{total_segments}")
                
            except Exception as e:
                print(f"\nError processing segment {segment_id}: {str(e)}")
                # Save progress even if this segment failed
                save_analyzed_segments(output_dir, analyzed_segments)
                continue
        
        video.close()
        print("\nVideo splitting and analysis complete.")
        return analyzed_segments
        
    except Exception as e:
        print(f"\nError during video splitting and analysis: {str(e)}")
        # Save whatever progress we made
        save_analyzed_segments(output_dir, analyzed_segments)
        raise
    finally:
        # Make sure we always close the video file
        if 'video' in locals():
            video.close()