FROM i386/debian:bookworm

WORKDIR /home

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libc6-dev \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install \
        setuptools \
        wheel \
        pytest \
        pelfy

ENV PATH="/opt/venv/bin:$PATH"