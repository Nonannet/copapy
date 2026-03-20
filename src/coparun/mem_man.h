#ifndef MEM_MAN_H
#define MEM_MAN_H

uint8_t *allocate_executable_memory(uint32_t num_bytes);
uint8_t *allocate_data_memory(uint32_t num_bytes);
uint8_t *allocate_buffer_memory(uint32_t num_bytes);
int mark_mem_executable(uint8_t *memory, uint32_t memory_len);
void deallocate_memory(uint8_t *memory, uint32_t memory_len);

#ifdef DATA_MEMORY_LEN
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wbuiltin-declaration-mismatch"
void memcpy(void *dest, const void *src, unsigned int n);
#pragma GCC diagnostic pop
#else
#include <string.h>
#endif

#endif /* MEM_MAN_H */
