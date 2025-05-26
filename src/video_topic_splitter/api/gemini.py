import os
import logging
import base64
import mimetypes
import time
from typing import Optional, Union, Dict, Any
from pathlib import Path
import google.generativeai as genai
from PIL import Image
import tempfile
import subprocess

logger = logging.getLogger(__name__)

class GeminiClient:
    """
    Client for interacting with Google's Gemini API.
    Handles text and multimodal (image/video) content with rate limiting.
    """

    def __init__(self, 
                 api_key: str, 
                 model_name: str = "gemini-1.5-flash", 
                 rate_limit_requests: int = 15,
                 rate_limit_period: int = 60,
                 retry_count: int = 3,
                 retry_delay: int = 5):
        """
        Initialize the Gemini client with API key and model selection.

        Args:
            api_key: Google API key for Gemini
            model_name: Name of the Gemini model to use (default: gemini-1.5-flash)
            rate_limit_requests: Maximum number of requests allowed in the rate limit period
            rate_limit_period: Time period in seconds for the rate limit
            retry_count: Number of times to retry a failed request
            retry_delay: Delay in seconds between retries
        """
        self.api_key = api_key
        self.model_name = model_name
        
        # Rate limiting parameters
        self.rate_limit_requests = rate_limit_requests
        self.rate_limit_period = rate_limit_period
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        
        # Rate limiting state
        self.request_timestamps = []
        
        # Configure the Gemini API
        genai.configure(api_key=self.api_key)
        
        # Get the model
        try:
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"Successfully initialized Gemini client with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}", exc_info=True)
            raise

    def _check_rate_limit(self):
        """
        Check if we're within rate limits and wait if necessary.
        
        Returns:
            True if the request can proceed, False if rate limit is exceeded
        """
        current_time = time.time()
        
        # Remove timestamps older than the rate limit period
        self.request_timestamps = [ts for ts in self.request_timestamps 
                                  if current_time - ts < self.rate_limit_period]
        
        # Check if we've hit the rate limit
        if len(self.request_timestamps) >= self.rate_limit_requests:
            oldest_timestamp = min(self.request_timestamps)
            sleep_time = self.rate_limit_period - (current_time - oldest_timestamp)
            
            if sleep_time > 0:
                logger.warning(f"Rate limit reached. Waiting {sleep_time:.2f} seconds before next request.")
                time.sleep(sleep_time)
                # After waiting, recursively check again
                return self._check_rate_limit()
        
        # Add current timestamp to the list
        self.request_timestamps.append(current_time)
        return True

    def analyze(self, prompt: str, media_path: Optional[Union[str, Image.Image]] = None) -> str:
        """
        Analyzes text or multimodal content using the configured Gemini model.
        
        Args:
            prompt: The text prompt to send to Gemini
            media_path: Optional path to an image or video file, or a PIL Image object
            
        Returns:
            The text response from the Gemini API
        """
        for attempt in range(self.retry_count + 1):
            try:
                # Check rate limit before making the request
                self._check_rate_limit()
                
                # Handle different input types
                if media_path is None:
                    # Text-only prompt
                    logger.info("Sending text-only prompt to Gemini API")
                    response = self.model.generate_content(prompt)
                    
                elif isinstance(media_path, Image.Image):
                    # Direct PIL Image object
                    logger.info("Sending image with prompt to Gemini API")
                    response = self.model.generate_content([prompt, media_path])
                    
                elif isinstance(media_path, str) and os.path.exists(media_path):
                    # File path provided
                    file_extension = os.path.splitext(media_path)[1].lower()
                    
                    # Check if it's a video file
                    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
                    if file_extension in video_extensions:
                        logger.info(f"Processing video file: {media_path}")
                        return self._process_video(prompt, media_path)
                        
                    # Handle as image file
                    else:
                        try:
                            logger.info(f"Loading image from path: {media_path}")
                            img = Image.open(media_path)
                            response = self.model.generate_content([prompt, img])
                        except Exception as img_err:
                            logger.error(f"Failed to process image file: {img_err}", exc_info=True)
                            return f"Analysis failed: Could not process image file: {str(img_err)}"
                else:
                    logger.error(f"Invalid media path or unsupported media type: {media_path}")
                    return "Analysis failed: Invalid media path or unsupported media type"
                    
                # Process the response
                if hasattr(response, 'text'):
                    return response.text
                else:
                    # Handle different response formats
                    logger.warning(f"Unexpected response format: {type(response)}")
                    return str(response)
                    
            except Exception as e:
                if attempt < self.retry_count:
                    wait_time = self.retry_delay * (attempt + 1)
                    logger.warning(f"Attempt {attempt+1} failed: {e}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {self.retry_count} retry attempts failed: {e}", exc_info=True)
                    return f"Analysis failed after {self.retry_count} attempts: {str(e)}"
        
        # This should not be reached due to the return in the last exception handler
        return "Analysis failed: Unknown error"

    def _process_video(self, prompt: str, video_path: str) -> str:
        """
        Process a video file for Gemini analysis by extracting frames.
        
        Args:
            prompt: The text prompt to send to Gemini
            video_path: Path to the video file
            
        Returns:
            The text response from the Gemini API
        """
        try:
            # Create a temporary directory for extracted frames
            with tempfile.TemporaryDirectory() as temp_dir:
                logger.info(f"Created temporary directory for video frames: {temp_dir}")
                
                # Extract frames using ffmpeg
                frame_rate = "0.5"  # Extract 1 frame every 2 seconds
                frames_path = os.path.join(temp_dir, "frame_%04d.jpg")
                
                ffmpeg_cmd = [
                    "ffmpeg", "-i", video_path, 
                    "-vf", f"fps={frame_rate}", 
                    "-q:v", "2",  # High quality
                    frames_path
                ]
                
                logger.info(f"Extracting frames with command: {' '.join(ffmpeg_cmd)}")
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"Failed to extract frames: {result.stderr}")
                    return f"Analysis failed: Could not extract video frames: {result.stderr}"
                
                # Get the extracted frames
                frames = sorted([os.path.join(temp_dir, f) for f in os.listdir(temp_dir) 
                                if f.startswith("frame_") and f.endswith(".jpg")])
                
                if not frames:
                    logger.error("No frames were extracted from the video")
                    return "Analysis failed: No frames could be extracted from the video"
                
                logger.info(f"Extracted {len(frames)} frames from video")
                
                # Select a subset of frames if there are too many
                MAX_FRAMES = 10  # Gemini has input token limits
                if len(frames) > MAX_FRAMES:
                    # Take evenly spaced frames
                    step = len(frames) // MAX_FRAMES
                    selected_frames = frames[::step][:MAX_FRAMES]
                else:
                    selected_frames = frames
                
                logger.info(f"Selected {len(selected_frames)} frames for analysis")
                
                # Load the frames as PIL Images
                images = []
                for frame_path in selected_frames:
                    try:
                        img = Image.open(frame_path)
                        images.append(img)
                    except Exception as img_err:
                        logger.warning(f"Could not load frame {frame_path}: {img_err}")
                
                if not images:
                    logger.error("Failed to load any frames as images")
                    return "Analysis failed: Could not load video frames as images"
                
                # Create the content for the API request
                content = [prompt]
                content.extend(images)
                
                # Check rate limit before making the request
                self._check_rate_limit()
                
                # Send to Gemini API with retry logic
                for attempt in range(self.retry_count + 1):
                    try:
                        logger.info(f"Sending {len(images)} video frames with prompt to Gemini API (attempt {attempt+1})")
                        response = self.model.generate_content(content)
                        
                        if hasattr(response, 'text'):
                            return response.text
                        else:
                            return str(response)
                    except Exception as e:
                        if attempt < self.retry_count:
                            wait_time = self.retry_delay * (attempt + 1)
                            logger.warning(f"Video analysis attempt {attempt+1} failed: {e}. Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"All {self.retry_count} video analysis retry attempts failed: {e}", exc_info=True)
                            return f"Video analysis failed after {self.retry_count} attempts: {str(e)}"
                
        except Exception as e:
            logger.error(f"Error processing video for Gemini analysis: {e}", exc_info=True)
            return f"Analysis failed: Error processing video: {str(e)}"