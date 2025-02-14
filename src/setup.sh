# SETUP.sh
#!/bin/bash

# Create directories
mkdir -p video_topic_splitter/api
mkdir -p video_topic_splitter/analysis
mkdir -p video_topic_splitter/processing/audio
mkdir -p video_topic_splitter/processing/ocr
mkdir -p video_topic_splitter/processing/video
mkdir -p video_topic_splitter/utils

# Initialize Python packages (create __init__.py files)
touch video_topic_splitter/api/__init__.py
touch video_topic_splitter/analysis/__init__.py
touch video_topic_splitter/processing/__init__.py
touch video_topic_splitter/processing/audio/__init__.py
touch video_topic_splitter/processing/ocr/__init__.py
touch video_topic_splitter/processing/video/__init__.py
touch video_topic_splitter/utils/__init__.py

echo "Directory structure created under video_topic_splitter/."