# Use an official NVIDIA CUDA runtime as a parent image
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Metadata
LABEL maintainer="Your Name <your.email@example.com>"
LABEL org.opencontainers.image.source="https://github.com/your-repo/video-topic-splitter"

# Set non-interactive frontend
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including Python, pip, FFmpeg, curl, and Tesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    ffmpeg \
    curl \
    nano \
    libxcb-glx0 \
    libxxf86vm-dev \
    libxcb-cursor0 \
    tesseract-ocr \
    tesseract-ocr-eng \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -ms /bin/bash -u 1001 -U vts
WORKDIR /home/vts
ENV PATH="/home/vts/.local/bin:${PATH}"

# Copy application code
COPY --chown=vts:vts src /home/vts/src
COPY --chown=vts:vts setup.py /home/vts/
COPY --chown=vts:vts requirements.txt /home/vts/

# Switch to non-root user for dependency installation
USER vts

# Install Python dependencies
RUN pip install --user --no-cache-dir -r requirements.txt

# Set the entrypoint
ENTRYPOINT ["/bin/bash"]