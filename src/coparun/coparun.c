#include <stdio.h>
#include <stdlib.h>
#include "runmem.h"
#include "mem_man.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <code_file>\n", argv[0]);
        return EXIT_FAILURE;
    }

    FILE *f = fopen(argv[1], "rb");
    if (!f) {
        perror("fopen");
        return EXIT_FAILURE;
    }

    /* Determine file size */
    if (fseek(f, 0, SEEK_END) != 0) {
        perror("fseek");
        fclose(f);
        return EXIT_FAILURE;
    }
    long sz = ftell(f);
    if (sz < 0) {
        perror("ftell");
        fclose(f);
        return EXIT_FAILURE;
    }
    rewind(f);

    if (sz == 0) {
        fprintf(stderr, "Error: File is empty\n");
        fclose(f);
        return EXIT_FAILURE;
    }

    uint8_t *file_buff = allocate_buffer_memory((uint32_t)sz);

    /* Read file into allocated memory */
    size_t nread = fread(file_buff, 1, (size_t)sz, f);
    fclose(f);
    if (nread != (size_t)sz) {
        perror("fread");
        return EXIT_FAILURE;
    }

    int ret = parse_commands(file_buff);

    if (ret == 2) {
        /* Dump code for debugging */
        if (argc != 3) {
            fprintf(stderr, "Usage: %s <code_file> <memory_dump_file>\n", argv[0]);
            return EXIT_FAILURE;
        }
        f = fopen(argv[2], "wb");
        fwrite(executable_memory, 1, (size_t)executable_memory_len, f);
        fclose(f);
    }

    free_memory();

    return ret < 0;
}
