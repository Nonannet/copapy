FROM debian:stable

WORKDIR /home

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc-13 \
    gcc-13-arm-linux-gnueabihf \
    gcc-13-mips-linux-gnu \
    gcc-13-mipsel-linux-gnu \
    gcc-13-riscv64-linux-gnu \
    gcc-13-aarch64-linux-gnu \
    gcc-13-multilib \
    python3
