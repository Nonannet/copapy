#ifndef RUNMEM_H
#define RUNMEM_H

#include <stdint.h>

/* Command opcodes used by the parser */
#define ALLOCATE_DATA    1
#define COPY_DATA        2
#define ALLOCATE_CODE    3
#define COPY_CODE        4
#define PATCH_FUNC       5
#define PATCH_OBJECT     6
#define ENTRY_POINT      7
#define RUN_PROG        64
#define READ_DATA       65
#define END_COM        256
#define FREE_MEMORY    257

/* Relocation types */
#define PATCH_RELATIVE_32 0

/* Memory blobs accessible by other translation units */
extern uint8_t *data_memory;
extern uint32_t data_memory_len;
extern uint8_t *executable_memory;
extern uint32_t executable_memory_len;
extern int data_offs;

/* Entry point type and variable */
typedef int (*entry_point_t)(void);
extern entry_point_t entr_point;


/* Command parser: takes a pointer to the command stream and returns
   an error flag (0 on success according to current code) */
int parse_commands(uint8_t *bytes);

/* Free program and data memory */
void free_memory();

#endif /* RUNMEM_H */