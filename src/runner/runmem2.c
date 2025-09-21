#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <string.h>
#include <stdint.h>

#define ALLOCATE_DATA 1
#define COPY_DATA 2
#define ALLOCATE_CODE 3
#define COPY_CODE 4
#define PATCH_FUNC 5
#define PATCH_OBJECT 6
#define SET_ENTR_POINT 64
#define END_PROG 255

#define PATCH_RELATIVE_32 0

uint8_t *data_memory;
uint8_t *executable_memory;
uint32_t executable_memory_len;
int (*entr_point)();

uint8_t *get_executable_memory(uint32_t num_bytes){
    // Allocate executable memory
    uint8_t *mem = (uint8_t*)mmap(NULL, num_bytes,
                         PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return mem;
}

uint8_t *get_data_memory(uint32_t num_bytes) {
    uint8_t *mem = (uint8_t*)mmap(NULL, num_bytes,
                     PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    //uint8_t *mem = (uint8_t*)malloc(num_bytes);
    return mem;
}

int mark_mem_executable(){
    if (mprotect(executable_memory, executable_memory_len, PROT_READ | PROT_EXEC) == -1) {
        perror("mprotect failed");
        return 0;
    }else{
        return 1;
    }
}

void patch_mem_32(uint8_t *patch_addr, int32_t value){
    int32_t *val_ptr = (int32_t*)patch_addr;
    *val_ptr = value;
}
    
int patch(uint8_t *patch_addr, uint32_t reloc_type, int32_t value){
    if (reloc_type == PATCH_RELATIVE_32){
        patch_mem_32(patch_addr, value);
    }else{
        printf("Not implemented");
        return 0;
    }
    return 1;
}  

int parse_commands(uint8_t *bytes){
    int32_t value;
    uint32_t command;
    uint32_t reloc_type;
    uint32_t offs;
    int data_offs;
    uint32_t size;
    int err_flag = 0;
    
    while(!err_flag){
        command = *(uint32_t*)bytes;
        bytes += 4;
        switch(command) {
        	case ALLOCATE_DATA:
                size = *(uint32_t*)bytes; bytes += 4;
                printf("ALLOCATE_DATA size=%i\n", size);
                data_memory = get_data_memory(size);
                break;
            
        	case COPY_DATA:
                offs = *(uint32_t*)bytes; bytes += 4;
                size = *(uint32_t*)bytes; bytes += 4;
                printf("COPY_DATA offs=%i size=%i\n", offs, size);
                memcpy(data_memory + offs, bytes, size); bytes += size;
                break;
            
        	case ALLOCATE_CODE:
                size = *(uint32_t*)bytes; bytes += 4;
                printf("ALLOCATE_CODE size=%i\n", size);
                executable_memory = get_executable_memory(size);
                executable_memory_len = size;
                //printf("# d %i  c %i  off %i\n", data_memory, executable_memory, data_offs);
                break;
            
            case COPY_CODE:
                offs = *(uint32_t*)bytes; bytes += 4;
                size = *(uint32_t*)bytes; bytes += 4;
                printf("COPY_CODE offs=%i size=%i\n", offs, size);
                memcpy(executable_memory + offs, bytes, size); bytes += size;
                break;
            
            case PATCH_FUNC:
                offs = *(uint32_t*)bytes; bytes += 4;
                reloc_type = *(uint32_t*)bytes; bytes += 4;
                value = *(int32_t*)bytes; bytes += 4;
                printf("PATCH_FUNC patch_offs=%i reloc_type=%i value=%i\n",
                    offs, reloc_type, value);
                patch(executable_memory + offs, reloc_type, value);
                break;
            
            case PATCH_OBJECT:
                offs = *(uint32_t*)bytes; bytes += 4;
                reloc_type = *(uint32_t*)bytes; bytes += 4;
                value = *(int32_t*)bytes; bytes += 4;
                printf("PATCH_OBJECT patch_offs=%i reloc_type=%i value=%i\n",
                    offs, reloc_type, value);
                data_offs = (int32_t)(data_memory - executable_memory);
                if (abs(data_offs) > 0x7FFFFFFF) {
                    perror("code and data memory to far apart");
                    return EXIT_FAILURE;
                }
                patch(executable_memory + offs, reloc_type, value + (int32_t)data_offs);
                //printf("> %i\n", data_offs);
                break;
            
            case SET_ENTR_POINT:
                uint32_t rel_entr_point = *(uint32_t*)bytes; bytes += 4;
                printf("SET_ENTR_POINT rel_entr_point=%i\n", rel_entr_point);
                entr_point = (int (*)())(executable_memory + rel_entr_point);    
                break;
            
            case END_PROG:
                printf("END_PROG\n");
                mark_mem_executable();
                int ret = entr_point();
                printf("Return value: %i\n", ret);
                err_flag = 1;
                break;
            
        	default:
                printf("Unknown command\n");
                err_flag = -1;
                break;
        }
    }
    return err_flag;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <binary_file>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Open the file
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) {
        perror("open");
        return EXIT_FAILURE;
    }

    // Get file size
    struct stat st;
    if (fstat(fd, &st) < 0) {
        perror("fstat");
        close(fd);
        return EXIT_FAILURE;
    }

    if (st.st_size == 0) {
        fprintf(stderr, "Error: File is empty\n");
        close(fd);
        return EXIT_FAILURE;
    }

    //uint8_t *file_buff = get_data_memory((uint32_t)st.st_size);
    uint8_t *file_buff = (uint8_t*)malloc((size_t)st.st_size);

    // Read file into allocated memory
    if (read(fd, file_buff, (long unsigned int)st.st_size) != st.st_size) {
        perror("read");
        close(fd);
        return EXIT_FAILURE;
    }
    close(fd);

    parse_commands(file_buff);

    munmap(executable_memory, executable_memory_len);
    return EXIT_SUCCESS;
}
