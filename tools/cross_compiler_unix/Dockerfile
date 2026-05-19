FROM debian:stable

WORKDIR /home

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc-13 \
    gcc-13-i686-linux-gnu \
    gcc-arm-none-eabi \
    gcc-13-mips-linux-gnu \
    gcc-13-mipsel-linux-gnu \
    gcc-13-riscv64-linux-gnu \
    gcc-13-aarch64-linux-gnu \
    gcc-13-multilib \
    python3 \
    git

RUN wget https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/2026.04.26/riscv32-musl-ubuntu-22.04-gcc.tar.xz && \
    tar -xf riscv32-musl-ubuntu-22.04-gcc.tar.xz && \
    rm riscv32-musl-ubuntu-22.04-gcc.tar.xz

ENV PATH=/home/riscv/bin:$PATH

COPY *.sh ./

RUN sh build_musl.sh
