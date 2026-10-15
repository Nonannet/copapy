/*
 * file: mem_man.c
 * Description: This file contains memory management functions for the coparun
 * runner, including allocation and deallocation of executable and data memory.
 * Depending of the target operating system or bare metal environment, it
 * handles memory management accordingly.
*/

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

#elif defined(__ZEPHYR__)

#include <zephyr/kernel.h>
#include <zephyr/init.h>
#include <zephyr/cache.h>

/* Zephyr implementations: code, data and buffer memory are taken from a
 * dedicated heap. Without MMU there is no per-allocation protection, the
 * heap must be located in a memory region that is not marked execute-never
 * by the MPU (see CONFIG_COPARUN_HEAP_SECTION in targets/zephyr/Kconfig). */

#define COPARUN_MEM_ALIGN 16

static uint8_t heap_memory[CONFIG_COPARUN_HEAP_SIZE] __aligned(8)
    __attribute__((section(CONFIG_COPARUN_HEAP_SECTION)));
static struct k_heap coparun_heap;

static int coparun_heap_init(void) {
    k_heap_init(&coparun_heap, heap_memory, sizeof(heap_memory));
    return 0;
}

SYS_INIT(coparun_heap_init, APPLICATION, CONFIG_APPLICATION_INIT_PRIORITY);

uint8_t *allocate_executable_memory(uint32_t num_bytes) {
    uint8_t *mem = (uint8_t*)k_heap_aligned_alloc(&coparun_heap, COPARUN_MEM_ALIGN, num_bytes, K_NO_WAIT);
    if (mem == NULL) {
        fprintf(stderr, "k_heap_aligned_alloc failed (executable)\n");
    }
    return mem;
}

uint8_t *allocate_data_memory(uint32_t num_bytes) {
    uint8_t *mem = (uint8_t*)k_heap_aligned_alloc(&coparun_heap, COPARUN_MEM_ALIGN, num_bytes, K_NO_WAIT);
    if (mem == NULL) {
        fprintf(stderr, "k_heap_aligned_alloc failed (data)\n");
    }
    return mem;
}

uint8_t *allocate_buffer_memory(uint32_t num_bytes) {
    return (uint8_t*)k_heap_aligned_alloc(&coparun_heap, COPARUN_MEM_ALIGN, num_bytes, K_NO_WAIT);
}

int mark_mem_executable(uint8_t *memory, uint32_t memory_len) {
    if (!memory || memory_len == 0) return 0;
    /* Required on cores with separate data and instruction caches (e.g.
       Cortex-M7), no-ops if the cache management is not enabled */
    sys_cache_data_flush_range(memory, memory_len);
    sys_cache_instr_invd_range(memory, memory_len);
    return 1;
}

void deallocate_memory(uint8_t *memory, uint32_t memory_len) {
    (void)memory_len;
    if (memory) k_heap_free(&coparun_heap, memory);
}

#else

#include <sys/mman.h>

/* POSIX implementations */

uint8_t *allocate_executable_memory(uint32_t num_bytes) {
    uint8_t *mem = (uint8_t*)mmap(NULL, num_bytes,
                         PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {
        perror("mmap failed (executable)");
        return NULL;
    }
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
    if (mem == MAP_FAILED) {
        perror("mmap failed (data)");
        return NULL;
    }
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
#if defined(__GNUC__)
    /* Required on architectures without coherent instruction cache (e.g. ARM, RISC-V) */
    __builtin___clear_cache((char*)memory, (char*)memory + memory_len);
#endif
    return 1;
}

void deallocate_memory(uint8_t *memory, uint32_t memory_len) {
    if (memory_len) munmap(memory, memory_len);
}

#endif
