/* Output and exit by ARM semihosting, used to run the example in QEMU.
   On real hardware replace it by e.g. a UART driver. */
#ifndef SEMIHOST_H
#define SEMIHOST_H

static inline int semihost_call(int op, const void *arg) {
    register int r0 __asm__("r0") = op;
    register const void *r1 __asm__("r1") = arg;
    __asm__ volatile("bkpt 0xAB" : "+r"(r0) : "r"(r1) : "memory");
    return r0;
}

static inline void semihost_write(const char *text) {
    semihost_call(0x04, text);  /* SYS_WRITE0 */
}

static inline void semihost_exit(int code) {
    const int block[2] = {0x20026, code};  /* ADP_Stopped_ApplicationExit */
    semihost_call(0x20, block);  /* SYS_EXIT_EXTENDED */
    for (;;);
}

#endif
