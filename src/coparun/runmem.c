#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "runmem.h"
#include "mem_man.h"

void patch(uint8_t *patch_addr, uint32_t patch_mask, int32_t value) {
    uint32_t *val_ptr = (uint32_t*)patch_addr;
    uint32_t original = *val_ptr;

    uint32_t shift_factor = patch_mask & -patch_mask;

    uint32_t new_value = (original & ~patch_mask) | (((uint32_t)value * shift_factor) & patch_mask);

    *val_ptr = new_value;
}

void patch_hi21(uint8_t *patch_addr, int32_t page_offset) {
    uint32_t instr = *(uint32_t *)patch_addr;

    // Split page_offset into immhi (upper 19 bits) and immlo (lower 2 bits)
    uint32_t immlo = page_offset & 0x3;        // bits[1:0]
    uint32_t immhi = (page_offset >> 2) & 0x7FFFF; // bits[20:2]

    // Clear previous imm fields: immhi (bits[23:5]) and immlo (bits[30:29])
    instr &= ~((0x7FFFFu << 5) | (0x3 << 29));

    // Set new immhi and immlo
    instr |= (immhi << 5) | (immlo << 29);

    *(uint32_t *)patch_addr = instr;
}

void patch_arm32_abs(uint8_t *patch_addr, uint32_t imm16)
{
    uint32_t instr = *((uint32_t *)patch_addr);

    // Split the 16-bit immediate into A1 MOVT fields
    uint32_t imm4  = (imm16 >> 12) & 0xF;
    uint32_t imm12 =  imm16        & 0xFFF;

    // Clear the immediate fields: imm4 (bits 19:16) and imm12 (bits 11:0)
    instr &= ~(uint32_t)((0xF << 16) | 0xFFF);

    // Set new immediate fields
    instr |= (imm4 << 16);
    instr |= imm12;

    *((uint32_t *)patch_addr) = instr;
}

void free_memory(runmem_t *context) {
    deallocate_memory(context->executable_memory, context->executable_memory_len);
    deallocate_memory(context->data_memory, context->data_memory_len);
    context->executable_memory_len = 0;
    context->data_memory_len = 0;
    context->executable_memory = NULL;
    context->data_memory = NULL;
    context->entr_point = NULL;
    context->data_offs = 0;
}

int update_data_offs(runmem_t *context) {
    if (context->data_memory && context->executable_memory &&
        (context->data_memory - context->executable_memory > 0x7FFFFFFF ||
         context->executable_memory - context->data_memory > 0x7FFFFFFF)) {
        perror("Error: code and data memory to far apart");
        return 0;
    }
    context->data_offs = (int)(context->data_memory - context->executable_memory);
    return 1;
}

int floor_div(int a, int b) {
    return a / b - ((a % b != 0) && ((a < 0) != (b < 0)));
}

int parse_commands(runmem_t *context, uint8_t *bytes) {
    int32_t value;
    uint32_t command;
    uint32_t patch_mask;
    int32_t patch_scale;
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
                context->data_memory = allocate_data_memory(size);
                context->data_memory_len = size;
                LOG("ALLOCATE_DATA size=%i mem_addr=%p\n", size, (void*)context->data_memory);
                if (!update_data_offs(context)) end_flag = -4;
                break;

            case COPY_DATA:
                offs = *(uint32_t*)bytes; bytes += 4;
                size = *(uint32_t*)bytes; bytes += 4;
                LOG("COPY_DATA offs=%i size=%i\n", offs, size);
                memcpy(context->data_memory + offs, bytes, size); bytes += size;
                break;

            case ALLOCATE_CODE:
                size = *(uint32_t*)bytes; bytes += 4;
                context->executable_memory = allocate_executable_memory(size);
                context->executable_memory_len = size;
                LOG("ALLOCATE_CODE size=%i mem_addr=%p\n", size, (void*)context->executable_memory);
                if (!update_data_offs(context)) end_flag = -4;
                break;

            case COPY_CODE:
                offs = *(uint32_t*)bytes; bytes += 4;
                size = *(uint32_t*)bytes; bytes += 4;
                LOG("COPY_CODE offs=%i size=%i\n", offs, size);
                memcpy(context->executable_memory + offs, bytes, size); bytes += size;
                break;

            case PATCH_FUNC:
                offs = *(uint32_t*)bytes; bytes += 4;
                patch_mask = *(uint32_t*)bytes; bytes += 4;
                patch_scale = *(int32_t*)bytes; bytes += 4;
                value = *(int32_t*)bytes; bytes += 4;
                LOG("PATCH_FUNC patch_offs=%i patch_mask=%#08x scale=%i value=%i\n",
                    offs, patch_mask, patch_scale, value);
                patch(context->executable_memory + offs, patch_mask, value / patch_scale);
                break;

            case PATCH_OBJECT:
                offs = *(uint32_t*)bytes; bytes += 4;
                patch_mask = *(uint32_t*)bytes; bytes += 4;
                patch_scale = *(int32_t*)bytes; bytes += 4;
                value = *(int32_t*)bytes; bytes += 4;
                LOG("PATCH_OBJECT patch_offs=%i patch_mask=%#08x scale=%i value=%i\n",
                    offs, patch_mask, patch_scale, value);
                patch(context->executable_memory + offs, patch_mask, value / patch_scale + context->data_offs / patch_scale);
                break;

            case PATCH_OBJECT_ABS:
                offs = *(uint32_t*)bytes; bytes += 4;
                patch_mask = *(uint32_t*)bytes; bytes += 4;
                patch_scale = *(int32_t*)bytes; bytes += 4;
                value = *(int32_t*)bytes; bytes += 4;
                LOG("PATCH_OBJECT_ABS patch_offs=%i patch_mask=%#08x scale=%i value=%i\n",
                    offs, patch_mask, patch_scale, value);
                patch(context->executable_memory + offs, patch_mask, value / patch_scale);
                break;

            case PATCH_OBJECT_REL:
                offs = *(uint32_t*)bytes; bytes += 4;
                bytes += 4;
                patch_scale = *(int32_t*)bytes; bytes += 4;
                value = *(int32_t*)bytes; bytes += 4;
                LOG("PATCH_OBJECT_REL patch_offs=%i patch_addr=%p scale=%i value=%i\n",
                    offs, (void*)(context->data_memory + value), patch_scale, value);
                *(void **)(context->executable_memory + offs) = context->data_memory + value;
                break;

            case PATCH_OBJECT_HI21:
                offs = *(uint32_t*)bytes; bytes += 4;
                bytes += 4;
                patch_scale = *(int32_t*)bytes; bytes += 4;
                value = *(int32_t*)bytes; bytes += 4;
                LOG("PATCH_OBJECT_HI21 patch_offs=%i scale=%i value=%i res_value=%i\n",
                    offs, patch_scale, value, floor_div(context->data_offs + value, patch_scale) - (int32_t)offs / patch_scale);
                patch_hi21(context->executable_memory + offs, floor_div(context->data_offs + value, patch_scale) - (int32_t)offs / patch_scale);
                break;

            case PATCH_OBJECT_ARM32_ABS:
                offs = *(uint32_t*)bytes; bytes += 4;
                patch_mask = *(uint32_t*)bytes; bytes += 4;
                patch_scale = *(int32_t*)bytes; bytes += 4;
                value = *(int32_t*)bytes; bytes += 4;
                LOG("PATCH_OBJECT_ARM32_ABS patch_offs=%i patch_mask=%#08x scale=%i value=%i imm16=%#04x\n",
                    offs, patch_mask, patch_scale, value, (uint32_t)((uintptr_t)(context->data_memory + value) & patch_mask) / (uint32_t)patch_scale);
                patch_arm32_abs(context->executable_memory + offs, (uint32_t)((uintptr_t)(context->data_memory + value) & patch_mask) / (uint32_t)patch_scale);
                break;

            case ENTRY_POINT:
                rel_entr_point = *(uint32_t*)bytes; bytes += 4;
                context->entr_point = (entry_point_t)(context->executable_memory + rel_entr_point);
                LOG("ENTRY_POINT rel_entr_point=%i\n", rel_entr_point); 
                mark_mem_executable(context->executable_memory, context->executable_memory_len);
                break;

            case RUN_PROG:
                LOG("RUN_PROG\n");
                {
                    int ret = context->entr_point();
                    (void)ret;
                    BLOG("Return value: %i\n", ret);
                }
                break;

            case READ_DATA:
                offs = *(uint32_t*)bytes; bytes += 4;
                size = *(uint32_t*)bytes; bytes += 4;
                BLOG("READ_DATA offs=%i size=%i data=", offs, size);
                for (uint32_t i = 0; i < size; i++) {
                    printf("%02X ", context->data_memory[offs + i]);
                }
                printf("\n");
                break;

            case FREE_MEMORY:
                LOG("FREE_MENORY\n");
                free_memory(context);
                break;

            case DUMP_CODE:
                LOG("DUMP_CODE\n");
                end_flag = 2;
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
