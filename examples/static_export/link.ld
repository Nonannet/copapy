/* Linker script for the QEMU machine mps2-an386 (Cortex-M4).
 *
 * COPAPY_CODE_ADDR and COPAPY_DATA_ADDR are passed by the Makefile (--defsym),
 * the copapy program is generated for the same addresses.
 */
ENTRY(Reset_Handler)

MEMORY
{
    FLASH (rx) : ORIGIN = 0x00000000, LENGTH = 4M
    RAM (rw)   : ORIGIN = 0x20000000, LENGTH = 4M
}

_estack = ORIGIN(RAM) + LENGTH(RAM);

SECTIONS
{
    .text :
    {
        KEEP(*(.isr_vector))
        *(.text*)
        *(.rodata*)
        . = ALIGN(4);
    } > FLASH

    .ARM.exidx : { *(.ARM.exidx*) } > FLASH

    _sidata = LOADADDR(.data);

    .data :
    {
        _sdata = .;
        *(.data*)
        . = ALIGN(4);
        _edata = .;
    } > RAM AT > FLASH

    .bss (NOLOAD) :
    {
        _sbss = .;
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
    } > RAM

    /* Copapy program: machine code, executed from flash */
    .copapy_code COPAPY_CODE_ADDR :
    {
        KEEP(*(.copapy_code))
    } > FLASH

    /* Copapy program: data memory, initialized by <name>_init */
    .copapy_data COPAPY_DATA_ADDR (NOLOAD) :
    {
        KEEP(*(.copapy_data))
    } > RAM
}
