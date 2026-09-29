/*
 * file: runmem.c
 * Description: This file contain the core functions of the runner
 * to receive data, code and patch instruction, does the patching
 * and jumps to the entry point of the copapy program
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "runmem.h"
#include "mem_man.h"

void patch_bitfield(uint8_t *patch_addr, uint32_t patch_mask, int32_t value) {
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

void patch_arm_movw_movt(uint8_t *patch_addr, uint32_t imm16)
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

void patch_thumb_movw_movt(uint8_t *patch_addr, uint32_t imm16)
{
    // Thumb MOVW (T3) / MOVT (T1) encoding
    
    uint16_t *instr16 = (uint16_t *)patch_addr;
    uint16_t first_half = instr16[0];
    uint16_t second_half = instr16[1];

    // Extract fields from imm16
    uint32_t imm4 = (imm16 >> 12) & 0xF;
    uint32_t i    = (imm16 >> 11) & 0x1;
    uint32_t imm3 = (imm16 >> 8) & 0x7;
    uint32_t imm8 = imm16 & 0xFF;

    // Clear bits
    first_half &= (uint16_t)(~(0x000F | (1 << 10)));
    second_half &= (uint16_t)(~(0x00FF | (0x7 << 12)));

    // Set new fields
    first_half |= (uint16_t)((imm4 << 0) | (i << 10));
    second_half |= (uint16_t)(imm8 | (imm3 << 12));

    instr16[0] = first_half;
    instr16[1] = second_half;
}

void patch_thumb_branch(uint8_t *patch_addr, int32_t offset)
{
    // Read the 32-bit instruction (two halfwords)
    uint16_t *instr16 = (uint16_t *)patch_addr;
    uint16_t first_half  = instr16[0];
    uint16_t second_half = instr16[1];

    // offset is the byte offset already shifted right by 1
    // (Thumb branch targets are halfword aligned)

    // Split into S, J1, J2, imm10, imm11
    uint32_t S     = (offset >> 23) & 0x1;
    uint32_t I1    = (offset >> 22) & 0x1;
    uint32_t I2    = (offset >> 21) & 0x1;
    uint32_t imm10 = (offset >> 11) & 0x3FF;
    uint32_t imm11 = offset & 0x7FF;

    // Re-encode J1 and J2
    uint32_t J1 = (~(I1 ^ S)) & 0x1;
    uint32_t J2 = (~(I2 ^ S)) & 0x1;

    // Clear old imm fields
    first_half  &= 0xF800;           // Keep upper 5 bits
    second_half &= 0xD000;           // Keep upper 5 bits

    // Set new imm fields
    first_half  |= (uint16_t)((S << 10) | imm10);
    second_half |= (uint16_t)((J1 << 13) | (J2 << 11) | imm11);

    // Write back
    instr16[0] = first_half;
    instr16[1] = second_half;
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

int apply_patch(runmem_t *context, uint32_t offs, int32_t value, uint32_t patch_mask,
                uint8_t encoding, uint8_t shift, uint8_t flags) {
    uint8_t *patch_addr = context->executable_memory + offs;
    uint8_t *base = (flags & PATCH_FLAG_DATA) ? context->data_memory : context->executable_memory;
    uintptr_t target = (uintptr_t)(base + value);  // S + A
    uintptr_t pc = (uintptr_t)patch_addr;          // P
    intptr_t result;

    if (flags & PATCH_FLAG_PAGE) {
        target &= ~(uintptr_t)0xFFF;
        pc &= ~(uintptr_t)0xFFF;
    }

    result = (flags & PATCH_FLAG_PC_REL) ? (intptr_t)(target - pc) : (intptr_t)target;
    result >>= shift;  // Arithmetic shift (floor division) for negative values

    LOG("PATCH patch_offs=%i value=%i mask=%#08x encoding=%i shift=%i flags=%#02x result=%#08x\n",
        offs, value, patch_mask, encoding, shift, flags, (uint32_t)result);

    switch (encoding) {
        case PATCH_ENC_BITFIELD:
            patch_bitfield(patch_addr, patch_mask, (int32_t)result);
            break;
        case PATCH_ENC_AARCH64_ADRP:
            patch_hi21(patch_addr, (int32_t)result);
            break;
        case PATCH_ENC_ARM_MOVW_MOVT:
            patch_arm_movw_movt(patch_addr, (uint32_t)result & 0xFFFF);
            break;
        case PATCH_ENC_THUMB_MOVW_MOVT:
            patch_thumb_movw_movt(patch_addr, (uint32_t)result & 0xFFFF);
            break;
        case PATCH_ENC_THUMB_BRANCH:
            patch_thumb_branch(patch_addr, (int32_t)result);
            break;
        default:
            LOG("Unknown patch encoding\n");
            return 0;
    }
    return 1;
}

int parse_commands(runmem_t *context, uint8_t *bytes) {
    int32_t value;
    uint32_t command;
    uint32_t patch_mask;
    uint8_t encoding, shift, flags;
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

            case PATCH:
                offs = *(uint32_t*)bytes; bytes += 4;
                value = *(int32_t*)bytes; bytes += 4;
                patch_mask = *(uint32_t*)bytes; bytes += 4;
                encoding = bytes[0];
                shift = bytes[1];
                flags = bytes[2];
                bytes += 4;
                if (!apply_patch(context, offs, value, patch_mask, encoding, shift, flags)) end_flag = -5;
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
                    int ret = call_entry_point(context->entr_point);
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
