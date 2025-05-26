# main.py
#!/usr/bin/env python3
"""
Entry point for the video segmentation and analysis tool.
"""

import logging
# os, GeminiClient, VideoProcessor were unused imports

# Assuming cli_main handles argument parsing and core logic execution
from .cli import main as cli_main

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    cli_main()
