#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef _WIN32
#include <windows.h>

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
