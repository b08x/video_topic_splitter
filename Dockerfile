# Use the linuxserver.io FFmpeg image as the base
FROM linuxserver/ffmpeg:amd64-6.1.1

# Metadata
LABEL maintainer="Your Name <your.email@example.com>"
LABEL org.opencontainers.image.source="https://github.com/your-repo/video-topic-splitter"

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
    bash-completion \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -ms /usr/bin/bash -u 1000 -U vts
    
WORKDIR /home/vts

# Set environment variables for the vts user
ENV HOME="/home/vts"
ENV PATH="/home/vts/.local/bin:${PATH}"

# Create a custom .bashrc for the vts user with color prompt and completions
RUN echo 'export PS1="\[\033[38;5;45m\]\u\[\033[0m\]@\[\033[38;5;208m\]\h\[\033[0m\]:\[\033[38;5;34m\]\w\[\033[0m\]\\$ "' > /home/vts/.bashrc && \
    echo 'export TERM=xterm-256color' >> /home/vts/.bashrc && \
    echo 'source /etc/bash_completion' >> /home/vts/.bashrc && \
    echo 'alias ls="ls --color=auto"' >> /home/vts/.bashrc && \
    echo 'alias ll="ls -la"' >> /home/vts/.bashrc && \
    echo 'alias grep="grep --color=auto"' >> /home/vts/.bashrc && \
    echo 'export HISTCONTROL=ignoreboth:erasedups' >> /home/vts/.bashrc && \
    echo 'export HISTSIZE=1000' >> /home/vts/.bashrc && \
    echo 'export HISTFILESIZE=2000' >> /home/vts/.bashrc && \
    echo 'shopt -s histappend' >> /home/vts/.bashrc && \
    chown vts:vts /home/vts/.bashrc

# Copy only requirements first to leverage Docker cache
COPY --chown=vts:vts requirements.txt /home/vts/

# Switch to vts user
USER vts

# Install dependencies as vts user
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the application code
COPY --chown=vts:vts src /home/vts/src
COPY --chown=vts:vts setup.py /home/vts/

# Install the application in development mode
RUN pip install --no-cache-dir -e .

# Set entrypoint (using -m for correct module resolution)
# ENTRYPOINT ["python3", "-m", "video_topic_splitter.cli"]
ENTRYPOINT ["/bin/bash"]
