FROM i386/ubuntu:24.04

WORKDIR /home

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libc6-dev-i386 \
    lib32stdc++-12-dev \
    python3 \
    python3-pip \
    python3-dev \
    && pip install setuptools wheel pytest pelfy
