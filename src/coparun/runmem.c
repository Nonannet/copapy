#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "runmem.h"
#include "mem_man.h"

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

/* Globals declared extern in runmem.h */
uint8_t *data_memory = NULL;
uint32_t data_memory_len = 0;
uint8_t *executable_memory = NULL;
uint32_t executable_memory_len = 0;
entry_point_t entr_point = NULL;
int data_offs = 0;

void patch_mem_32(uint8_t *patch_addr, int32_t value) {
    int32_t *val_ptr = (int32_t*)patch_addr;
    *val_ptr = value;
}
    
int patch(uint8_t *patch_addr, uint32_t reloc_type, int32_t value) {
    if (reloc_type == PATCH_RELATIVE_32) {
        patch_mem_32(patch_addr, value);
    }else{
        LOG("Not implemented");
        return 0;
    }
    return 1;
}

void free_memory() {
    deallocate_memory(executable_memory, executable_memory_len);
    deallocate_memory(data_memory, data_memory_len);
    executable_memory_len = 0;
    data_memory_len = 0;
}

int update_data_offs() {
    if (data_memory && executable_memory && (data_memory - executable_memory > 0x7FFFFFFF || executable_memory - data_memory > 0x7FFFFFFF)) {
        perror("Error: code and data memory to far apart");
        return 0;
    }
    data_offs = (int)(data_memory - executable_memory);
    return 1;
}

int parse_commands(uint8_t *bytes) {
    int32_t value;
    uint32_t command;
    uint32_t reloc_type;
    uint32_t offs;
    uint32_t size;
    int end_flag = 0;
    uint32_t rel_entr_point = 0;
    
    while(!end_flag) {
        command = *(uint32_t*)bytes;
        bytes += 4;
        switch(command) {
            case ALLOCATE_DATA:
                size = *(uint32_t*)bytes; bytes += 4;
                data_memory = allocate_data_memory(size);
                data_memory_len = size;
                LOG("ALLOCATE_DATA size=%i mem_addr=%p\n", size, (void*)data_memory);
                if (!update_data_offs()) end_flag = -4;
                break;
            
            case COPY_DATA:
                offs = *(uint32_t*)bytes; bytes += 4;
                size = *(uint32_t*)bytes; bytes += 4;
                LOG("COPY_DATA offs=%i size=%i\n", offs, size);
                memcpy(data_memory + offs, bytes, size); bytes += size;
                break;
            
            case ALLOCATE_CODE:
                size = *(uint32_t*)bytes; bytes += 4;
                executable_memory = allocate_executable_memory(size);
                executable_memory_len = size;
                LOG("ALLOCATE_CODE size=%i mem_addr=%p\n", size, (void*)executable_memory);
                //LOG("# d %i  c %i  off %i\n", data_memory, executable_memory, data_offs);
                if (!update_data_offs()) end_flag = -4;
                break;
            
            case COPY_CODE:
                offs = *(uint32_t*)bytes; bytes += 4;
                size = *(uint32_t*)bytes; bytes += 4;
                LOG("COPY_CODE offs=%i size=%i\n", offs, size);
                memcpy(executable_memory + offs, bytes, size); bytes += size;
                break;
            
            case PATCH_FUNC:
                offs = *(uint32_t*)bytes; bytes += 4;
                reloc_type = *(uint32_t*)bytes; bytes += 4;
                value = *(int32_t*)bytes; bytes += 4;
                LOG("PATCH_FUNC patch_offs=%i reloc_type=%i value=%i\n",
                    offs, reloc_type, value);
                patch(executable_memory + offs, reloc_type, value);
                break;
            
            case PATCH_OBJECT:
                offs = *(uint32_t*)bytes; bytes += 4;
                reloc_type = *(uint32_t*)bytes; bytes += 4;
                value = *(int32_t*)bytes; bytes += 4;
                LOG("PATCH_OBJECT patch_offs=%i reloc_type=%i value=%i\n",
                    offs, reloc_type, value);
                patch(executable_memory + offs, reloc_type, value + data_offs);
                break;

            case ENTRY_POINT:
                LOG("ENTRY_POINT rel_entr_point=%i\n", rel_entr_point);
                rel_entr_point = *(uint32_t*)bytes; bytes += 4;
                entr_point = (entry_point_t)(executable_memory + rel_entr_point);  
                mark_mem_executable(executable_memory, executable_memory_len);
                break;
            
            case RUN_PROG:
                LOG("RUN_PROG\n");
                int ret = entr_point();
                BLOG("Return value: %i\n", ret);
                break;
            
            case READ_DATA:
                offs = *(uint32_t*)bytes; bytes += 4;
                size = *(uint32_t*)bytes; bytes += 4;
                BLOG("READ_DATA offs=%i size=%i data=", offs, size);
                for (uint32_t i = 0; i < size; i++) {
                    printf("%02X ", data_memory[offs + i]);
                }
                printf("\n");
                break;

            case FREE_MEMORY:
                free_memory();
                break;

            case END_COM:
                LOG("END_COM\n");
                end_flag = 1;
                break;
            
            default:
                LOG("Unknown command\n");
                end_flag = -1;
                break;
        }
    }
    return end_flag;
}
