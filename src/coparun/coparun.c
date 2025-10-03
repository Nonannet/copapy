#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include "runmem.h"

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

    free_memory();

    return EXIT_SUCCESS;
}
