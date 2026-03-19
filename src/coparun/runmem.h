/**
 * @file runmem.h
 * @brief Header file for runmem.c, which contains core functions of
 * the runner to receive data, code and patch instructions, perform
 * patching, and jump to the entry point of the copapy program.
 */

#ifndef RUNMEM_H
#define RUNMEM_H

#include <stdint.h>

#ifdef DATA_MEMORY_ADDR
    #define PRINTF(...)
#else
    #include <stdio.h>
    #define PRINTF(...) printf(__VA_ARGS__)
#endif

#ifdef ENABLE_LOGGING
    #define LOG(...) printf(__VA_ARGS__)
    #define BLOG(...) printf(__VA_ARGS__)
#elif ENABLE_BASIC_LOGGING
    #define LOG(...)
    #define BLOG(...) printf(__VA_ARGS__)
#else
    #define LOG(...)
    #define BLOG(...)
#endif

/* Command opcodes used by the parser */
#define ALLOCATE_DATA     0x040001
#define COPY_DATA         0x080002
#define ALLOCATE_CODE     0x040003
#define COPY_CODE         0x080004
#define PATCH_FUNC        0x101000
#define PATCH_FUNC_ARM32_THM   0x101005
#define PATCH_OBJECT      0x102000
#define PATCH_OBJECT_HI21 0x102001
#define PATCH_OBJECT_ABS  0x102002
#define PATCH_OBJECT_REL  0x102003
#define PATCH_OBJECT_ARM32_ABS 0x102004
#define PATCH_OBJECT_ARM32_ABS_THM 0x102006
#define ENTRY_POINT       0x040007
#define RUN_PROG          0x000040
#define READ_DATA         0x080041
#define END_COM           0x000100
#define FREE_MEMORY       0x000101
#define DUMP_CODE         0x000102

/* Entry point type */
typedef int (*entry_point_t)(void);

/* rx_state */
#define RX_STATE_IDLE    0
#define RX_STATE_HEADER  1
#define RX_STATE_DATA    2

/* state_flag */
#define STATE_FLAG_NONE 0
#define STATE_FLAG_END_COM 1
#define STATE_FLAG_DUMP_CODE 2
#define STATE_FLAG_UNKNOWN_COMMAND -1
#define STATE_FLAG_MEM_DIST -4   //code and data memory to far apart

/* Struct for run-time memory state */
typedef struct runmem_s {
    uint8_t *data_memory;            // Pointer to data memory
    uint32_t data_memory_len;        // Length of data memory
    uint8_t *executable_memory;      // Pointer to executable memory
    uint32_t executable_memory_len;  // Length of executable memory
    int data_offs;                   // Offset of data memory relative to executable memory
    entry_point_t entr_point;        // Entry point function pointer
    int32_t rx_state;               // State for receiving commands (idle, header, data)
    int32_t state_flag;              // Flag for result/error state
    uint8_t *data_src;
    uint8_t *data_dest;
    uint32_t data_size;
} runmem_t;

/* Command parser: takes a pointer to the command stream and returns
   an error flag (0 on success according to current code) */
int parse_commands(runmem_t *context, uint8_t *bytes, uint32_t lengths);

/* Free program and data memory */
void free_memory(runmem_t *context);

#endif /* RUNMEM_H */
