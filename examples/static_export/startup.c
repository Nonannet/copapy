/* Minimal startup code for a Cortex-M4 with FPU */
#include <stdint.h>
#include "semihost.h"

extern uint32_t _estack, _sidata, _sdata, _edata, _sbss, _ebss;

int main(void);

void Reset_Handler(void) {
    /* Enable the FPU (CP10 and CP11 full access): the copapy code uses it */
    *(volatile uint32_t *)0xE000ED88 |= 0xFu << 20;
    __asm__ volatile("dsb\n isb");

    const uint32_t *src = &_sidata;
    for (uint32_t *dst = &_sdata; dst < &_edata;) *dst++ = *src++;
    for (uint32_t *dst = &_sbss; dst < &_ebss;) *dst++ = 0;

    semihost_exit(main());
}

void Default_Handler(void) {
    semihost_write("Unexpected exception\n");
    semihost_exit(2);
}

__attribute__((section(".isr_vector"), used))
static void *const vector_table[16] = {
    &_estack,
    Reset_Handler,
    Default_Handler,  /* NMI */
    Default_Handler,  /* HardFault */
    Default_Handler,  /* MemManage */
    Default_Handler,  /* BusFault */
    Default_Handler,  /* UsageFault */
    0, 0, 0, 0,
    Default_Handler,  /* SVCall */
    Default_Handler,  /* DebugMon */
    0,
    Default_Handler,  /* PendSV */
    Default_Handler,  /* SysTick */
};
