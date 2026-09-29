/**
 * @file runmem.h
 * @brief Header file for runmem.c, which contains core functions of
 * the runner to receive data, code and patch instructions, perform
 * patching, and jump to the entry point of the copapy program.
 */

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
#define PATCH             0x1000
#define ENTRY_POINT       7
#define RUN_PROG         64
#define READ_DATA        65
#define END_COM         256
#define FREE_MEMORY     257
#define DUMP_CODE       258

/* PATCH command arguments:
 *   uint32 offs      address of the patched instruction relative to code memory
 *   int32  value     target address (S + A) relative to code or data memory
 *   uint32 mask      bit mask of the instruction field (PATCH_ENC_BITFIELD only)
 *   uint8  encoding  PATCH_ENC_*
 *   uint8  shift     right shift applied to the calculated value
 *   uint8  flags     PATCH_FLAG_*
 *   uint8  reserved
 *
 * The runner calculates the value to insert as:
 *   S = (flags & DATA ? data_memory : executable_memory) + value
 *   P = executable_memory + offs
 *   with PAGE: S and P rounded down to 4 KiB pages
 *   result = ((flags & PC_REL) ? S - P : S) >> shift
 */

/* Patch encodings: how the result is inserted into the instruction */
#define PATCH_ENC_BITFIELD        0  /* 32 bit word, result placed at the lowest set bit of mask */
#define PATCH_ENC_AARCH64_ADRP    1  /* AArch64 ADRP immhi:immlo (21 bit) */
#define PATCH_ENC_ARM_MOVW_MOVT   2  /* ARM MOVW/MOVT (A1) imm4:imm12 (16 bit) */
#define PATCH_ENC_THUMB_MOVW_MOVT 3  /* Thumb MOVW/MOVT (T3/T1) imm4:i:imm3:imm8 (16 bit) */
#define PATCH_ENC_THUMB_BRANCH    4  /* Thumb B.W/BL (T4/T1) S:J1:J2:imm10:imm11 (24 bit) */

/* Patch flags: how the result is calculated */
#define PATCH_FLAG_DATA   0x01  /* value is relative to data memory (else code memory) */
#define PATCH_FLAG_PC_REL 0x02  /* subtract address of the patched instruction */
#define PATCH_FLAG_PAGE   0x04  /* round target and instruction address down to 4 KiB pages */

/* Entry point type */
typedef int (*entry_point_t)(void);

#ifdef _WIN64
/* Assembly wrapper to preserve RSI/RDI when calling System V ABI code from Microsoft x64 ABI */
extern int x86_64_abi_shim(entry_point_t entry_point);

static inline int call_entry_point(entry_point_t fp) {
    return x86_64_abi_shim(fp);
}
#else
static inline int call_entry_point(entry_point_t fp) {
    return fp();
}
#endif

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