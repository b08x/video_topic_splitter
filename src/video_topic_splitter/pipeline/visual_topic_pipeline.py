#!/usr/bin/env python3
"""Pipeline integration for visual topic analysis.

This module provides functions to integrate visual analysis with topic modeling,
creating a comprehensive pipeline that analyzes both visual content and transcript
text to produce enriched video segmentation.
"""

import logging
import os
import json
from typing import Dict, List, Tuple, Any, Optional

from ..analysis.visual_analysis import analyze_scenes, load_analyzed_scenes
from ..analysis.visual_topic_modeling import process_transcript_with_visuals
from ..api.gemini import GeminiClient
from ..constants import CHECKPOINTS
from ..project import save_checkpoint, load_checkpoint

# Define logger at the module level before it's used
logger = logging.getLogger(__name__)

# Try different import paths for transcript_processing
try:
    # Try the direct import first
    from ..processing.transcript.transcript_processing import load_transcript_sentences
except ImportError:
    # Alternative import path if the module is elsewhere
    try:
        from ..analysis.transcript_processing import load_transcript_sentences
    except ImportError:
        # If still not found, try another common location
        try:
            from ..transcript_processing import load_transcript_sentences
        except ImportError:
            # If still not found, define a placeholder function to avoid runtime errors
            logger.error("Could not import load_transcript_sentences. Defining placeholder.")
            
            def load_transcript_sentences(path):
                """Placeholder function for loading transcript sentences."""
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)

def prepare_visual_frames_for_topic_modeling(
    scene_analysis_results: List[Dict]
) -> List[Dict]:
    """Converts scene analysis results into a format suitable for topic modeling.
    
    Args:
        scene_analysis_results (List[Dict]): Results from visual scene analysis.
        
    Returns:
        List[Dict]: List of frame dictionaries with timestamps and descriptions.
    """
    logger.info("Preparing visual frames for topic modeling...")
    visual_frames = []
    
    for scene in scene_analysis_results:
        scene_id = scene.get("scene_id")
        start_time = scene.get("start_time", 0.0)
        end_time = scene.get("end_time", 0.0)
        
        # Get frame analyses from this scene
        frame_analyses = scene.get("frame_analyses", [])
        
        for frame_idx, frame_analysis in enumerate(frame_analyses):
            # Skip frames with errors
            if "error" in frame_analysis:
                continue
                
            # Calculate approximate timestamp for this frame within the scene
            if len(frame_analyses) > 1:
                # Distribute frames evenly across scene duration
                frame_time = start_time + (end_time - start_time) * (frame_idx / (len(frame_analyses) - 1))
            else:
                # If only one frame, use the middle of the scene
                frame_time = (start_time + end_time) / 2
                
            # Create frame dictionary
            frame_dict = {
                "scene_id": scene_id,
                "frame_idx": frame_idx,
                "timestamp": frame_time,
                "description": frame_analysis.get("gemini_analysis", ""),
                "frame_path": frame_analysis.get("frame_path", ""),
                "software_detections": frame_analysis.get("software_detections", [])
            }
            
            visual_frames.append(frame_dict)
    
    # Sort frames by timestamp
    visual_frames.sort(key=lambda x: x.get("timestamp", 0.0))
    
    logger.info(f"Prepared {len(visual_frames)} visual frames for topic modeling")
    return visual_frames

def run_visual_topic_pipeline(
    input_video: str,
    project_path: str,
    gemini_api_key: str,
    scene_boundaries: Optional[List[Tuple[float, float]]] = None,
    software_list: Optional[List[str]] = None,
    ocr_lang: str = "eng",
    frames_per_scene: int = 3,  # Increased for better visual context
    frame_format: str = "jpg",
    compression_quality: int = 90,
    register: str = "gen-ai",
    force_reanalysis: bool = False
) -> Dict:
    """Runs the complete visual topic analysis pipeline.
    
    This function orchestrates the entire process of:
    1. Analyzing video scenes visually
    2. Processing transcript text
    3. Combining visual and textual analysis for topic modeling
    
    Args:
        input_video (str): Path to the input video file.
        project_path (str): Path to the project directory.
        gemini_api_key (str): API key for Gemini.
        scene_boundaries (Optional[List[Tuple[float, float]]], optional): List of scene 
            boundary tuples (start_time, end_time). Defaults to None.
        software_list (Optional[List[str]], optional): List of software names to detect.
            Defaults to None.
        ocr_lang (str, optional): Language for OCR. Defaults to "eng".
        frames_per_scene (int, optional): Number of frames to extract per scene.
            Defaults to 3.
        frame_format (str, optional): Format for extracted frames. Defaults to "jpg".
        compression_quality (int, optional): Compression quality for frames.
            Defaults to 90.
        register (str, optional): Analysis domain/register. Defaults to "gen-ai".
        force_reanalysis (bool, optional): Whether to force reanalysis of already
            processed data. Defaults to False.
            
    Returns:
        Dict: Dictionary containing the combined analysis results.
    """
    logger.info(f"Starting visual topic pipeline for video: {input_video}")
    
    # Check if visual topic analysis is already complete
    visual_topic_results_path = os.path.join(project_path, "visual_topic_analysis_results.json")
    if os.path.exists(visual_topic_results_path) and not force_reanalysis:
        logger.info(f"Loading existing visual topic analysis from: {visual_topic_results_path}")
        try:
            with open(visual_topic_results_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load existing visual topic results: {e}. Proceeding with analysis.")
    
    # Initialize Gemini client
    gemini_client = GeminiClient(api_key=gemini_api_key)
    
    # Step 1: Run or load visual scene analysis
    analysis_results_dir = os.path.join(project_path, "scene_analysis")
    os.makedirs(analysis_results_dir, exist_ok=True)
    
    scene_analysis_checkpoint = load_checkpoint(project_path, CHECKPOINTS["SCENE_ANALYSIS_COMPLETE"])
    
    if scene_analysis_checkpoint and not force_reanalysis:
        logger.info("Loading existing scene analysis results...")
        scene_analysis_results = load_analyzed_scenes(analysis_results_dir)
        if not scene_analysis_results:
            logger.warning("Failed to load existing scene analysis results. Running analysis again.")
            scene_analysis_results = analyze_scenes(
                input_video=input_video,
                scene_boundaries=scene_boundaries,
                project_path=project_path,
                gemini_client=gemini_client,
                software_list=software_list,
                ocr_lang=ocr_lang,
                frames_per_scene=frames_per_scene,
                frame_format=frame_format,
                compression_quality=compression_quality,
                register=register
            )
    else:
        logger.info("Running visual scene analysis...")
        scene_analysis_results = analyze_scenes(
            input_video=input_video,
            scene_boundaries=scene_boundaries,
            project_path=project_path,
            gemini_client=gemini_client,
            software_list=software_list,
            ocr_lang=ocr_lang,
            frames_per_scene=frames_per_scene,
            frame_format=frame_format,
            compression_quality=compression_quality,
            register=register
        )
    
    # Step 2: Load transcript sentences
    transcript_path = os.path.join(project_path, "transcript", "transcript_sentences.json")
    if not os.path.exists(transcript_path):
        logger.error(f"Transcript sentences not found at: {transcript_path}")
        raise FileNotFoundError(f"Transcript sentences file not found: {transcript_path}")
    
    logger.info(f"Loading transcript sentences from: {transcript_path}")
    transcript_sentences = load_transcript_sentences(transcript_path)
    
    if not transcript_sentences:
        logger.error("No transcript sentences loaded. Cannot proceed with topic modeling.")
        raise ValueError("Empty transcript sentences list")
    
    # Step 3: Prepare visual frames for topic modeling
    visual_frames = prepare_visual_frames_for_topic_modeling(scene_analysis_results)
    
    if not visual_frames:
        logger.warning("No visual frames prepared. Topic modeling will rely solely on transcript.")
    
    # Step 4: Run visual topic modeling
    logger.info("Running visual topic modeling...")
    visual_topic_results = process_transcript_with_visuals(
        transcript_sentences=transcript_sentences,
        visual_frames=visual_frames,
        project_path=project_path,
        register=register
    )
    
    # Step 5: Save combined results
    combined_results_path = os.path.join(project_path, "combined_analysis_results.json")
    logger.info(f"Saving combined analysis results to: {combined_results_path}")
    
    combined_results = {
        "visual_topic_analysis": visual_topic_results,
        "scene_analysis": {
            "total_scenes": len(scene_analysis_results),
            "total_frames_analyzed": sum(len(scene.get("frame_analyses", [])) for scene in scene_analysis_results)
        },
        "metadata": {
            "input_video": input_video,
            "frames_per_scene": frames_per_scene,
            "register": register,
            "ocr_language": ocr_lang
        }
    }
    
    try:
        with open(combined_results_path, 'w', encoding='utf-8') as f:
            json.dump(combined_results, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save combined results: {e}")
    
    # Save final checkpoint
    save_checkpoint(
        project_path,
        CHECKPOINTS["VISUAL_TOPIC_PIPELINE_COMPLETE"],
        {
            "combined_results_path": combined_results_path,
            "visual_topic_results_path": visual_topic_results_path,
            "scene_analysis_path": os.path.join(analysis_results_dir, "scene_analysis_results.json")
        }
    )
    
    logger.info("Visual topic pipeline complete.")
    return combined_results


def extract_key_insights(combined_results: Dict) -> Dict:
    """Extracts key insights from the combined analysis results.
    
    This function processes the combined results to extract the most important
    insights, such as major topics, key visual elements, and significant moments.
    
    Args:
        combined_results (Dict): The combined analysis results.
        
    Returns:
        Dict: Dictionary containing extracted key insights.
    """
    logger.info("Extracting key insights from combined analysis results...")
    
    insights = {
        "major_topics": [],
        "key_visual_elements": [],
        "significant_moments": [],
        "software_usage": []
    }
    
    # Extract visual topic analysis
    visual_topic_analysis = combined_results.get("visual_topic_analysis", {})
    
    # Get topics
    topics = visual_topic_analysis.get("topics", [])
    visual_topics = visual_topic_analysis.get("visual_topics", [])
    segments = visual_topic_analysis.get("segments", [])
    
    # Process topics
    for topic in topics:
        topic_name = topic.get("topic", "Unknown")
        keywords = topic.get("words", [])
        
        if topic_name != "Unknown" and keywords:
            insights["major_topics"].append({
                "name": topic_name,
                "keywords": keywords[:5]  # Top 5 keywords
            })
    
    # Process visual topics
    for topic in visual_topics:
        topic_name = topic.get("topic", "Unknown")
        keywords = topic.get("words", [])
        
        if topic_name != "Unknown" and keywords:
            insights["key_visual_elements"].append({
                "name": topic_name,
                "keywords": keywords[:5]  # Top 5 keywords
            })
    
    # Find significant moments (segments with high confidence or clear topics)
    for segment in segments:
        # Check if this segment has a clear topic and visual topic
        if (segment.get("dominant_topic", "Unknown") != "Unknown" and 
            segment.get("visual_topic", "Unknown") != "Unknown"):
            
            insights["significant_moments"].append({
                "time_range": f"{segment.get('start_time', 0):.2f}s - {segment.get('end_time', 0):.2f}s",
                "text_topic": segment.get("dominant_topic", "Unknown"),
                "visual_topic": segment.get("visual_topic", "Unknown"),
                "visual_summary": segment.get("visual_summary", "")[:100] + "..." if len(segment.get("visual_summary", "")) > 100 else segment.get("visual_summary", "")
            })
    
    # Extract software usage from scene analysis
    scene_analysis = combined_results.get("scene_analysis", {})
    if "scene_analysis_results" in combined_results:
        for scene in combined_results.get("scene_analysis_results", []):
            detected_software = scene.get("detected_software", [])
            if detected_software:
                start_time = scene.get("start_time", 0)
                end_time = scene.get("end_time", 0)
                
                for software in detected_software:
                    # Check if this software is already in our list
                    existing = next((item for item in insights["software_usage"] if item["name"] == software), None)
                    
                    if existing:
                        # Add this time range to existing entry
                        existing["time_ranges"].append(f"{start_time:.2f}s - {end_time:.2f}s")
                    else:
                        # Create new entry
                        insights["software_usage"].append({
                            "name": software,
                            "time_ranges": [f"{start_time:.2f}s - {end_time:.2f}s"]
                        })
    
    # Limit the number of significant moments
    if len(insights["significant_moments"]) > 10:
        # Sort by duration (longer segments might be more significant)
        insights["significant_moments"].sort(
            key=lambda x: float(x["time_range"].split(" - ")[1][:-1]) - float(x["time_range"].split(" - ")[0][:-1]),
            reverse=True
        )
        insights["significant_moments"] = insights["significant_moments"][:10]
    
    logger.info("Key insights extraction complete.")
    return insights
