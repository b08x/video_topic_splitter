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
    nano \
    python3 \
    python3-pip \
    python3-dev build-essential \
    libxcb-glx0 \
    libxxf86vm-dev \
    libxcb-cursor0 \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/vts

ENV PATH="$HOME/.local/bin:${PATH}"
# Switch to non-root user after installation
# Copy application code with correct permissions
COPY --chown=vts:vts src /home/vts/src
COPY --chown=vts:vts setup.py /home/vts/
COPY --chown=vts:vts requirements.txt /home/vts/

# Install dependencies as user
RUN pip install --user --no-cache-dir .

# Set entrypoint (using -m for correct module resolution)
# ENTRYPOINT ["python3", "-m", "video_topic_splitter.cli"]
ENTRYPOINT ["/bin/bash"]
