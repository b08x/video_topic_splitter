"""Thumbnail generation and management utilities."""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

class ThumbnailManager:
    """Manages thumbnail generation and selection for video scenes."""
    
    def __init__(self, project_path: str):
        """
        Initialize the ThumbnailManager.
        
        Args:
            project_path: Path to the project directory
        """
        self.project_path = project_path
        self.thumbnails_dir = os.path.join(project_path, "thumbnails")
        os.makedirs(self.thumbnails_dir, exist_ok=True)
        
    def generate_thumbnails(self, 
                           video_path: str, 
                           scene_boundaries: List[Tuple[float, float]], 
                           frames_per_scene: int = 3,
                           format: str = "jpg",
                           quality: int = 90) -> Dict[int, List[str]]:
        """
        Generate thumbnails for each scene in the video.
        
        Args:
            video_path: Path to the video file
            scene_boundaries: List of (start_time, end_time) tuples in seconds
            frames_per_scene: Number of frames to extract per scene
            format: Image format (jpg, png)
            quality: Compression quality for JPEG (0-100)
            
        Returns:
            Dictionary mapping scene index to list of thumbnail paths
        """
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return {}
            
        # Import here to avoid circular imports
        from ..processing.video.frame_extraction import extract_frames_at_timestamps
        
        thumbnails_by_scene = {}
        
        for i, (start_time, end_time) in enumerate(scene_boundaries):
            scene_duration = end_time - start_time
            
            # Calculate evenly spaced timestamps for this scene
            if frames_per_scene == 1:
                # Just take the middle frame
                timestamps = [start_time + (scene_duration / 2)]
            else:
                # Distribute frames evenly
                timestamps = [
                    start_time + (j * scene_duration / (frames_per_scene - 1))
                    for j in range(frames_per_scene)
                ]
                # Ensure we don't go beyond the end time
                timestamps = [min(t, end_time - 0.1) for t in timestamps]
            
            # Create scene directory
            scene_dir = os.path.join(self.thumbnails_dir, f"scene_{i}")
            os.makedirs(scene_dir, exist_ok=True)
            
            # Extract frames
            thumbnail_paths = []
            for j, timestamp in enumerate(timestamps):
                output_path = os.path.join(scene_dir, f"frame_{j}.{format}")
                try:
                    success = extract_frames_at_timestamps(
                        video_path=video_path,
                        timestamps=[timestamp],
                        output_dir=scene_dir,
                        output_template=f"frame_{j}.{format}",
                        format=format,
                        quality=quality
                    )
                    if success and os.path.exists(output_path):
                        thumbnail_paths.append(output_path)
                    else:
                        logger.warning(f"Failed to extract frame at {timestamp}s for scene {i}")
                except Exception as e:
                    logger.error(f"Error extracting frame at {timestamp}s: {str(e)}")
            
            if thumbnail_paths:
                thumbnails_by_scene[i] = thumbnail_paths
        
        return thumbnails_by_scene
    
    def select_best_thumbnail(self, 
                             thumbnails: List[str], 
                             criteria: str = "clarity") -> Optional[str]:
        """
        Select the best thumbnail from a list based on specified criteria.
        
        Args:
            thumbnails: List of thumbnail paths
            criteria: Selection criteria ('clarity', 'brightness', 'contrast')
            
        Returns:
            Path to the best thumbnail or None if selection fails
        """
        if not thumbnails:
            return None
            
        if len(thumbnails) == 1:
            return thumbnails[0]
            
        scores = []
        
        for path in thumbnails:
            try:
                img = cv2.imread(path)
                if img is None:
                    logger.warning(f"Could not read image: {path}")
                    scores.append(-1)
                    continue
                    
                if criteria == "clarity":
                    # Laplacian variance as a measure of clarity/sharpness
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    score = cv2.Laplacian(gray, cv2.CV_64F).var()
                elif criteria == "brightness":
                    # Average brightness
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    score = hsv[:, :, 2].mean()
                elif criteria == "contrast":
                    # Standard deviation of grayscale as a measure of contrast
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    score = gray.std()
                else:
                    # Default to clarity
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    score = cv2.Laplacian(gray, cv2.CV_64F).var()
                    
                scores.append(score)
            except Exception as e:
                logger.error(f"Error processing thumbnail {path}: {str(e)}")
                scores.append(-1)
        
        if not scores or max(scores) <= 0:
            return thumbnails[0]  # Default to first if all failed
            
        best_index = scores.index(max(scores))
        return thumbnails[best_index]
    
    def get_scene_thumbnails(self, 
                            video_path: str, 
                            scene_boundaries: List[Tuple[float, float]],
                            frames_per_scene: int = 3,
                            format: str = "jpg",
                            quality: int = 90,
                            selection_criteria: str = "clarity") -> Dict[int, str]:
        """
        Generate and select the best thumbnail for each scene.
        
        Args:
            video_path: Path to the video file
            scene_boundaries: List of (start_time, end_time) tuples in seconds
            frames_per_scene: Number of frames to extract per scene
            format: Image format (jpg, png)
            quality: Compression quality for JPEG (0-100)
            selection_criteria: Criteria for selecting the best thumbnail
            
        Returns:
            Dictionary mapping scene index to best thumbnail path
        """
        all_thumbnails = self.generate_thumbnails(
            video_path=video_path,
            scene_boundaries=scene_boundaries,
            frames_per_scene=frames_per_scene,
            format=format,
            quality=quality
        )
        
        best_thumbnails = {}
        for scene_idx, thumbnails in all_thumbnails.items():
            best = self.select_best_thumbnail(thumbnails, criteria=selection_criteria)
            if best:
                best_thumbnails[scene_idx] = best
                
        return best_thumbnails