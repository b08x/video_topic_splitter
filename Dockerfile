# Use the linuxserver.io FFmpeg image as the base
FROM linuxserver/ffmpeg:amd64-6.1.1

# Metadata
LABEL maintainer="Your Name <your.email@example.com>"
LABEL org.opencontainers.image.source="https://github.com/your-repo/video-topic-splitter"

USER root

RUN groupadd -g 1001 vts
RUN useradd -ms /bin/bash -u 1000 -g 1001 vts

# Install Python and other system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev build-essential \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/vts

# Copy application code (including setup.py and data/logos)
COPY src /home/vts/src
COPY setup.py /home/vts/
COPY requirements.txt /home/vts/

# Set ownership and install dependencies as root
RUN chown -R vts:vts /home/vts && \
    pip install --no-cache-dir .

# Switch to non-root user after installation
USER vts

# Set entrypoint (using -m for correct module resolution)
ENTRYPOINT ["python", "-m", "video_topic_splitter.cli"]