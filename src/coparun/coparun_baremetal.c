/*
 * file: coparun_baremetal.c
 * Description: Runner for bare metal targets without file system, e.g. for
 * running copapy programs in qemu-system (tools/tricore/run_qemu.sh).
 *
 * The command stream is placed in memory at COMMAND_BUFFER_ADDR by an emulator
 * or debugger (e.g. qemu -device loader). All output written to stdout/stderr
 * is captured in coparun_output. When coparun_done is reached, the output
 * and the patched code (on DUMP_CODE) can be read back by a debugger with a
 * breakpoint on coparun_done. Afterwards the return code is written to
 * EXIT_DEVICE_ADDR, which terminates qemu with it as exit code.
 *
 * Build with -DCOPARUN_BARE_METAL
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "runmem.h"

#ifndef COMMAND_BUFFER_ADDR
#define COMMAND_BUFFER_ADDR 0xA1300000  /* qemu tricore_testboard external RAM */
#endif

#ifndef EXIT_DEVICE_ADDR
#define EXIT_DEVICE_ADDR 0xF0000000  /* qemu tricore_testboard test device */
#endif

#ifndef OUTPUT_BUFFER_SIZE
#define OUTPUT_BUFFER_SIZE 0x40000
#endif

/* Captured output and state are global to be accessible by the debugger */
char coparun_output[OUTPUT_BUFFER_SIZE];
volatile uint32_t coparun_output_len = 0;
runmem_t targ;

/* System call used by newlib for writing to stdout/stderr */
int write(int fd, const void *buf, size_t len) {
    (void)fd;
    for (size_t i = 0; i < len && coparun_output_len < OUTPUT_BUFFER_SIZE; i++) {
        coparun_output[coparun_output_len++] = ((const char*)buf)[i];
    }
    return (int)len;
}

/* Breakpoint location for the debugger */
__attribute__((noinline)) void coparun_done(int ret) {
    __asm__ volatile("" : : "r"(ret) : "memory");
}

__attribute__((noreturn)) static void terminate(int ret) {
    fflush(stdout);
    fflush(stderr);
    coparun_done(ret);
    *(volatile uint32_t*)EXIT_DEVICE_ADDR = (uint32_t)ret;
    for (;;);
}

void _exit(int ret) {
    terminate(ret);
}

int main(void) {
    memset(&targ, 0, sizeof(targ));

    int ret = parse_commands(&targ, (uint8_t*)COMMAND_BUFFER_ADDR);

    terminate(ret < 0);
    return 0;
}
