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
    python3-venv \
    git \
    wget \
    xz-utils \
    unzip \
    libexpat1 \
    libzstd1 \
    libtinfo6 \
    liblzma5

# RISC-V 32 bit cross compiler (not available as debian package)
RUN wget https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/2026.04.26/riscv32-musl-ubuntu-22.04-gcc.tar.xz && \
    tar -xf riscv32-musl-ubuntu-22.04-gcc.tar.xz && \
    rm riscv32-musl-ubuntu-22.04-gcc.tar.xz

ENV PATH=/home/riscv/bin:$PATH

# TriCore cross compiler with newlib, tricore-elf-gdb and qemu-system-tricore
# (not available as debian package)
RUN wget https://github.com/NoMore201/tricore-gcc-toolchain/releases/download/11.3.1-20250101/tricore-gcc-11.3.1-20250101-linux.zip && \
    mkdir -p /opt/tricore && \
    unzip -q tricore-gcc-11.3.1-20250101-linux.zip -d /opt/tricore && \
    rm tricore-gcc-11.3.1-20250101-linux.zip

ENV PATH=$PATH:/opt/tricore/bin

COPY *.sh ./

RUN sh build_musl.sh
