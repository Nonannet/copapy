/*
 * file: mem_man.c
 * Description: This file contains memory management functions for the coparun
 * runner, including allocation and deallocation of executable and data memory.
 * Depending of the target operating system or bare metal environment, it
 * handles memory management accordingly.
*/

#include <stdint.h>


#if defined DATA_MEMORY_ADDR || defined EXECUTABLE_MEMORY_ADDR
/* Bare metal implementations */
#if not defined(EXECUTABLE_MEMORY_ADDR) || not defined(DATA_MEMORY_ADDR)
    #error "For bare metal, you must define DATA_MEMORY_ADDR and DATA_EXECUTABLE_MEMORY_ADDR."
#endif

uint8_t *allocate_executable_memory(uint32_t num_bytes) {
    return (uint8_t*)EXECUTABLE_MEMORY_ADDR;
}

uint8_t *allocate_data_memory(uint32_t num_bytes) {
    return (uint8_t*)DATA_MEMORY_ADDR;
}

int mark_mem_executable(uint8_t *memory, uint32_t memory_len) {
    /* No-op for bare metal */
    return 1;
}

void deallocate_memory(uint8_t *memory, uint32_t memory_len) {
    /* No-op for bare metal */
}

void memcpy(void *dest, const void *src, size_t n) {
    uint8_t *d = (uint8_t*)dest;
    const uint8_t *s = (const uint8_t*)src;
    for (size_t i = 0; i < n; i++) {
        d[i] = s[i];
    }
    return 0;
}


#elif defined _WIN32
#include <windows.h>
#include <string.h>
#include <stdio.h>

/* Windows implementations */

uint8_t *allocate_executable_memory(uint32_t num_bytes) {
    uint8_t *mem = (uint8_t*)VirtualAlloc(NULL, (SIZE_T)num_bytes,
                                          MEM_RESERVE | MEM_COMMIT,
                                          PAGE_READWRITE);
    if (mem == NULL) {
        fprintf(stderr, "VirtualAlloc failed (executable): %lu\n", GetLastError());
    }
    return mem;
}

uint8_t *allocate_data_memory(uint32_t num_bytes) {
    /* Allocate RW memory that can later be made executable. */
    uint8_t *mem = (uint8_t*)VirtualAlloc(NULL, (SIZE_T)num_bytes,
                                          MEM_RESERVE | MEM_COMMIT,
                                          PAGE_READWRITE);
    if (mem == NULL) {
        fprintf(stderr, "VirtualAlloc failed (data): %lu\n", GetLastError());
    }
    return mem;
}

uint8_t *allocate_buffer_memory(uint32_t num_bytes) {
    return (uint8_t*)malloc((size_t)num_bytes);
}

int mark_mem_executable(uint8_t *memory, uint32_t memory_len) {
    if (!memory || memory_len == 0) return 0;
    DWORD oldProtect = 0;
    if (!VirtualProtect((LPVOID)memory, (SIZE_T)memory_len, PAGE_EXECUTE_READ, &oldProtect)) {
        fprintf(stderr, "VirtualProtect failed: %lu\n", GetLastError());
        return 0;
    }
    return 1;
}

void deallocate_memory(uint8_t *memory, uint32_t memory_len) {
    if (!memory) return;
    if (memory_len) {
        VirtualFree((LPVOID)memory, 0, MEM_RELEASE);
    } else {
        free(memory);
    }
}

#else

#include <sys/mman.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* POSIX implementations */

uint8_t *allocate_executable_memory(uint32_t num_bytes) {
    uint8_t *mem = (uint8_t*)mmap(NULL, num_bytes,
                         PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return mem;
}

uint8_t *allocate_data_memory(uint32_t num_bytes) {
    /*
    Malloc can not be used since it may return a memory region too far apart
    from the executable memory yielded by mmap for relative 32 bit addressing.

    uint8_t *mem = (uint8_t*)malloc(num_bytes);
    */
    uint8_t *mem = (uint8_t*)mmap(NULL, num_bytes,
                     PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return mem;
}

uint8_t *allocate_buffer_memory(uint32_t num_bytes) {
    return (uint8_t*)malloc((size_t)num_bytes);
}

int mark_mem_executable(uint8_t *memory, uint32_t memory_len) {
    if (mprotect(memory, memory_len, PROT_READ | PROT_EXEC) == -1) {
        perror("mprotect failed");
        return 0;
    }
    return 1;
}

void deallocate_memory(uint8_t *memory, uint32_t memory_len) {
    if (memory_len) munmap(memory, memory_len);
}

#endif
