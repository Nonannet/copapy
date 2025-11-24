FROM python:3.11-slim

WORKDIR /home

RUN apt-get update && apt-get install -y build-essential && \
    pip install setuptools wheel pytest pelfy
