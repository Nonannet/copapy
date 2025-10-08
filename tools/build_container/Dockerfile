FROM debian:stable

WORKDIR /home

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc-12 \
    gcc-12-arm-linux-gnueabihf \
    gcc-12-mips-linux-gnu \
    gcc-12-mipsel-linux-gnu \
    gcc-12-riscv64-linux-gnu \
    gcc-12-aarch64-linux-gnu \
    gcc-12-multilib