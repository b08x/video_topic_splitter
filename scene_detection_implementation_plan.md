# Implementation Plan for `--extract-scenes` Feature

This feature will replace the current video splitting approach with scene detection and extract JPG images instead of video segments, while integrating with the existing topic modeling.

## 1. Overview of Changes

- Add PySceneDetect to project dependencies
- Add `--extract-scenes` flag to CLI
- Create new scene detection module
- Modify visual analysis pipeline to use scene detection when enabled
- Update core processing to pass the flag through the pipeline
- Update checkpoints system for scene detection

## 2. Detailed Implementation Steps

### 2.1. Add PySceneDetect to Dependencies
- Add `scenedetect` to `requirements.txt`

### 2.2. Add Command-Line Argument
- Modify `cli.py` to add the `--extract-scenes` flag

### 2.3. Create Scene Detection Module
- Create `processing/video/scene_detection.py` with functions for:
  - Detecting scenes using PySceneDetect's content detector
  - Extracting representative frames from each scene
  - Saving frames as JPG images
  - Using sensible defaults for scene detection parameters

### 2.4. Modify Visual Analysis Pipeline
- Update `analysis/visual_analysis.py` to use scene detection when the flag is enabled
- Modify the `split_and_analyze_video` function to handle scene-based processing

### 2.5. Update Core Processing
- Modify `core.py` to pass the extract_scenes flag through the processing pipeline

### 2.6. Update Checkpoints System
- Add a new checkpoint in `constants.py` for scene detection

## 3. Technical Implementation Details

### 3.1. Scene Detection Implementation

```python
def detect_scenes(video_path, min_scene_len=1.0):
    """
    Detect scenes in a video using PySceneDetect.
    
    Args:
        video_path: Path to the video file
        min_scene_len: Minimum scene length in seconds
        
    Returns:
        List of scene boundaries (start_time, end_time)
    """
    # Use PySceneDetect's content detector with adaptive threshold
    # Extract scene boundaries as timestamps
    # Apply minimum scene length filter
    # Return list of scene boundaries
```

### 3.2. Frame Extraction

```python
def extract_scene_frames(video_path, scene_boundaries, output_dir):
    """
    Extract representative frames from each scene.
    
    Args:
        video_path: Path to the video file
        scene_boundaries: List of scene boundaries (start_time, end_time)
        output_dir: Directory to save extracted frames
        
    Returns:
        List of paths to extracted frames
    """
    # For each scene, extract a representative frame
    # Save frame as JPG in the output directory
    # Return list of frame paths
```

### 3.3. Integration with Topic Modeling

- The extracted scene frames will be analyzed using the existing visual analysis pipeline
- Topic modeling will be applied to the transcript as before
- The results will combine scene information with topic information

## 4. User Experience

When a user runs the command with `--extract-scenes`:

1. The video will be processed for audio and transcription as before
2. Instead of splitting the video into segments based on topics, scenes will be detected
3. Representative frames will be extracted from each scene and saved as JPGs
4. These frames will be analyzed for visual content
5. The results will combine scene information with topic modeling from the transcript

## 5. Benefits

- Faster processing by avoiding full video segment extraction
- More accurate visual representation by using natural scene boundaries
- Reduced storage requirements by saving only JPG images instead of video segments
- Maintained integration with existing topic modeling