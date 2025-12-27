#ifndef RUNMEM_H
#define RUNMEM_H

#include <stdint.h>

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
#define ALLOCATE_DATA     1
#define COPY_DATA         2
#define ALLOCATE_CODE     3
#define COPY_CODE         4
#define PATCH_FUNC        0x1000
#define PATCH_OBJECT      0x2000
#define PATCH_OBJECT_HI21 0x2001
#define PATCH_OBJECT_ABS  0x2002
#define PATCH_OBJECT_REL  0x2003
#define PATCH_OBJECT_ARM32_ABS 0x2004
#define ENTRY_POINT       7
#define RUN_PROG         64
#define READ_DATA        65
#define END_COM         256
#define FREE_MEMORY     257
#define DUMP_CODE       258

/* Entry point type */
typedef int (*entry_point_t)(void);

/* Struct for run-time memory state */
typedef struct runmem_s {
    uint8_t *data_memory;            // Pointer to data memory
    uint32_t data_memory_len;        // Length of data memory
    uint8_t *executable_memory;      // Pointer to executable memory
    uint32_t executable_memory_len;  // Length of executable memory
    int data_offs;                   // Offset of data memory relative to executable memory
    entry_point_t entr_point;        // Entry point function pointer
} runmem_t;

/* Command parser: takes a pointer to the command stream and returns
   an error flag (0 on success according to current code) */
int parse_commands(runmem_t *context, uint8_t *bytes);

/* Free program and data memory */
void free_memory(runmem_t *context);

#endif /* RUNMEM_H */