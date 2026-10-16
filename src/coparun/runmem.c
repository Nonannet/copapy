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
#include <time.h>
#include "runmem.h"
#include "mem_man.h"

/* 32 bit instruction access; memcpy since instructions might be only
   2 byte aligned (e.g. RISC-V with compressed instructions) */
static uint32_t read_u32(const uint8_t *addr) {
    uint32_t value;
    memcpy(&value, addr, sizeof(value));
    return value;
}

static void write_u32(uint8_t *addr, uint32_t value) {
    memcpy(addr, &value, sizeof(value));
}

/* Reads the next 32 bit word of the command stream and advances the position */
static uint32_t next_u32(uint8_t **bytes) {
    uint32_t value = read_u32(*bytes);
    *bytes += 4;
    return value;
}

void patch_bitfield(uint8_t *patch_addr, uint32_t patch_mask, int32_t value) {
    uint32_t original = read_u32(patch_addr);

    uint32_t shift_factor = patch_mask & -patch_mask;

    uint32_t new_value = (original & ~patch_mask) | (((uint32_t)value * shift_factor) & patch_mask);

    write_u32(patch_addr, new_value);
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

void patch_riscv_s_type(uint8_t *patch_addr, int32_t offset)
{
    // imm[11:5] -> bits[31:25], imm[4:0] -> bits[11:7]
    uint32_t imm = (uint32_t)offset;
    uint32_t instr = read_u32(patch_addr) & ~((0x7Fu << 25) | (0x1Fu << 7));
    instr |= ((imm >> 5) & 0x7F) << 25 | (imm & 0x1F) << 7;
    write_u32(patch_addr, instr);
}

void patch_riscv_b_type(uint8_t *patch_addr, int32_t offset)
{
    // imm[12] -> bit[31], imm[10:5] -> bits[30:25], imm[4:1] -> bits[11:8], imm[11] -> bit[7]
    uint32_t imm = (uint32_t)offset;
    uint32_t instr = read_u32(patch_addr) & ~((0x7Fu << 25) | (0x1Fu << 7));
    instr |= ((imm >> 12) & 0x1) << 31 | ((imm >> 5) & 0x3F) << 25 |
             ((imm >> 1) & 0xF) << 8 | ((imm >> 11) & 0x1) << 7;
    write_u32(patch_addr, instr);
}

void patch_riscv_j_type(uint8_t *patch_addr, int32_t offset)
{
    // imm[20] -> bit[31], imm[10:1] -> bits[30:21], imm[11] -> bit[20], imm[19:12] -> bits[19:12]
    uint32_t imm = (uint32_t)offset;
    uint32_t instr = read_u32(patch_addr) & 0xFFFu;
    instr |= ((imm >> 20) & 0x1) << 31 | ((imm >> 1) & 0x3FF) << 21 |
             ((imm >> 11) & 0x1) << 20 | ((imm >> 12) & 0xFF) << 12;
    write_u32(patch_addr, instr);
}

void patch_riscv_cb_type(uint8_t *patch_addr, int32_t offset)
{
    // 16 bit instruction: imm[8] -> bit[12], imm[4:3] -> bits[11:10],
    // imm[7:6] -> bits[6:5], imm[2:1] -> bits[4:3], imm[5] -> bit[2]
    uint32_t imm = (uint32_t)offset;
    uint16_t instr;
    memcpy(&instr, patch_addr, sizeof(instr));
    instr = (uint16_t)((instr & ~((0x7u << 10) | (0x1Fu << 2))) |
                       ((imm >> 8) & 0x1) << 12 | ((imm >> 3) & 0x3) << 10 |
                       ((imm >> 6) & 0x3) << 5 | ((imm >> 1) & 0x3) << 3 | ((imm >> 5) & 0x1) << 2);
    memcpy(patch_addr, &instr, sizeof(instr));
}

void patch_riscv_cj_type(uint8_t *patch_addr, int32_t offset)
{
    // 16 bit instruction: imm[11|4|9:8|10|6|7|3:1|5] -> bits[12:2]
    uint32_t imm = (uint32_t)offset;
    uint16_t instr;
    memcpy(&instr, patch_addr, sizeof(instr));
    instr = (uint16_t)((instr & ~(0x7FFu << 2)) |
                       ((imm >> 11) & 0x1) << 12 | ((imm >> 4) & 0x1) << 11 |
                       ((imm >> 8) & 0x3) << 9 | ((imm >> 10) & 0x1) << 8 |
                       ((imm >> 6) & 0x1) << 7 | ((imm >> 7) & 0x1) << 6 |
                       ((imm >> 1) & 0x7) << 3 | ((imm >> 5) & 0x1) << 2);
    memcpy(patch_addr, &instr, sizeof(instr));
}

void free_memory(runmem_t *context) {
    runmem_stop(context);
    deallocate_memory(context->executable_memory, context->executable_memory_len);
    deallocate_memory(context->data_memory, context->data_memory_len);
    context->executable_memory_len = 0;
    context->data_memory_len = 0;
    context->executable_memory = NULL;
    context->data_memory = NULL;
    context->entr_point = NULL;
    context->data_offs = 0;
    context->tasks = NULL;
    context->task_count = 0;
}

static void run_copies(runmem_t *context, uint32_t offs, uint32_t count) {
    copy_t *copies = (copy_t *)(context->data_memory + offs);
    RUNMEM_LOCK();
    for (uint32_t i = 0; i < count; i++) {
        memcpy(context->data_memory + copies[i].dst, context->data_memory + copies[i].src, copies[i].size);
    }
    RUNMEM_UNLOCK();
}

static void read_inputs(runmem_t *context, const task_t *task) {
    const io_t *io = (const io_t *)(context->data_memory + task->io_in_offs);
    for (uint32_t i = 0; i < task->io_in_count; i++) {
        runmem_io_read(io[i].channel, context->data_memory + io[i].addr);
    }
}

static void write_outputs(runmem_t *context, const task_t *task) {
    const io_t *io = (const io_t *)(context->data_memory + task->io_out_offs);
    for (uint32_t i = 0; i < task->io_out_count; i++) {
        runmem_io_write(io[i].channel, context->data_memory + io[i].addr);
    }
}

static int run_task(runmem_t *context, task_t *task) {
    task->runs++;
    return call_entry_point((entry_point_t)(context->executable_memory + task->entry));
}

void tasks_publish(runmem_t *context) {
    for (uint32_t i = 0; i < context->task_count; i++) {
        task_t *task = &context->tasks[i];
        if (task->period && task->countdown == 0 && task->pending) {
            run_copies(context, task->publish_offs, task->publish_count);
            write_outputs(context, task);
            task->pending = 0;
        }
    }
}

/*
 * Latches copies of all released tasks and runs the jobs of the released tasks.
 * Returns a bitmask of the released tasks (bit 0 = task 0, bit 1 = task 1, ...).
 */
uint32_t tasks_latch(runmem_t *context) {
    uint32_t released = 0;
    uint32_t now = runmem_timestamp();
    for (uint32_t i = 0; i < context->task_count && i < 32; i++) {
        task_t *task = &context->tasks[i];
        if (!task->period) continue;
        if (task->countdown == 0) {
            if (task->busy) {
                task->overruns++;
            } else {
                task->release_time = now;
                read_inputs(context, task);
                run_copies(context, task->latch_offs, task->latch_count);
                task->busy = 1;
                released |= 1u << i;
            }
            task->countdown = task->period;
        }
        task->countdown--;
    }
    return released;
}

/* Runs a job of a task and updates its statistics */
static int measured_job(runmem_t *context, task_t *task) {
    uint32_t start = runmem_timestamp();
    uint32_t cpu_start = runmem_cpu_time();
    int ret = run_task(context, task);
    uint32_t exec = runmem_to_ns(runmem_cpu_time() - cpu_start);
    uint32_t delay = runmem_to_ns(start - task->release_time);
    uint32_t response = runmem_to_ns(runmem_timestamp() - task->release_time);
    uint32_t sum_lo = task->exec_sum_lo + exec;

    task->exec_last = exec;
    if (exec > task->exec_max) task->exec_max = exec;
    if (sum_lo < exec) task->exec_sum_hi++;
    task->exec_sum_lo = sum_lo;
    if (delay > task->delay_max) task->delay_max = delay;
    if (response > task->response_max) task->response_max = response;
    return ret;
}

void task_job(runmem_t *context, uint32_t index) {
    task_t *task = &context->tasks[index];
    measured_job(context, task);
    RUNMEM_LOCK();
    task->pending = 1;
    task->busy = 0;
    RUNMEM_UNLOCK();
}

void tasks_release(runmem_t *context) {
    /* Latch copies of all released tasks before the first task runs */
    uint32_t released = tasks_latch(context);
    for (uint32_t i = 0; i < 32; i++) {
        if (released & (1u << i)) task_job(context, i);
    }
}

#ifndef __ZEPHYR__
/* Real-time execution is not supported by this runner */
int runmem_start(runmem_t *context, uint32_t tick_us) {
    (void)context;
    (void)tick_us;
    return 0;
}

void runmem_stop(runmem_t *context) {
    (void)context;
}

void runmem_write_data(const uint8_t *data, uint32_t size) {
    fwrite(data, 1, size, stdout);
}

/* Time in ns (wall time, also used as CPU time) */
uint32_t runmem_timestamp(void) {
#if defined(_WIN32) || defined(__linux__)
    struct timespec ts;
    if (timespec_get(&ts, TIME_UTC)) return (uint32_t)ts.tv_sec * 1000000000u + (uint32_t)ts.tv_nsec;
#endif
    return 0;
}

uint32_t runmem_cpu_time(void) {
    return runmem_timestamp();
}

uint32_t runmem_to_ns(uint32_t delta) {
    return delta;
}

/* No device I/O channels */
void runmem_io_read(uint32_t channel, uint8_t *value) {
    (void)channel;
    (void)value;
}

void runmem_io_write(uint32_t channel, const uint8_t *value) {
    (void)channel;
    (void)value;
}

void runmem_io_list(void) {
}
#endif

int update_data_offs(runmem_t *context) {
    // Data memory is not allocated if a program has no data
    if (!context->data_memory || !context->executable_memory) return 1;
    if (context->data_memory - context->executable_memory > 0x7FFFFFFF ||
        context->executable_memory - context->data_memory > 0x7FFFFFFF) {
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
        case PATCH_ENC_RISCV_S_TYPE:
            patch_riscv_s_type(patch_addr, (int32_t)result);
            break;
        case PATCH_ENC_RISCV_B_TYPE:
            patch_riscv_b_type(patch_addr, (int32_t)result);
            break;
        case PATCH_ENC_RISCV_J_TYPE:
            patch_riscv_j_type(patch_addr, (int32_t)result);
            break;
        case PATCH_ENC_RISCV_CB_TYPE:
            patch_riscv_cb_type(patch_addr, (int32_t)result);
            break;
        case PATCH_ENC_RISCV_CJ_TYPE:
            patch_riscv_cj_type(patch_addr, (int32_t)result);
            break;
        default:
            LOG("Unknown patch encoding\n");
            return 0;
    }
    return 1;
}

void runmem_init(runmem_t *context) {
    context->data_memory = NULL;
    context->data_memory_len = 0;
    context->executable_memory = NULL;
    context->executable_memory_len = 0;
    context->data_offs = 0;
    context->entr_point = NULL;
    context->rx_state = RX_STATE_IDLE;
    context->state_flag = STATE_FLAG_NONE;
    context->data_dest = NULL;
    context->data_size = 0;
    context->tasks = NULL;
    context->task_count = 0;
    context->tasks_published = 0;
    context->running = 0;
}

/* Size of the arguments following the command word in bytes, without the
   payload of COPY_DATA and COPY_CODE */
static uint32_t command_header_size(uint32_t command) {
    switch(command) {
        case ALLOCATE_DATA:
        case ALLOCATE_CODE:
        case ENTRY_POINT:
        case RUN_TASK:
        case STEP_TASKS:
        case START_TASKS:
            return 4;
        case COPY_DATA:
        case COPY_CODE:
        case READ_DATA:
        case READ_MEM:
        case TASK_TABLE:
            return 8;
        case PATCH:
            return 16;
        default:
            return 0;
    }
}

int parse_commands(runmem_t *context, uint8_t *bytes, uint32_t lengths) {
    int32_t value;
    uint32_t command;
    uint32_t patch_mask;
    uint8_t encoding, shift, flags;
    uint32_t offs;
    uint32_t size;
    uint32_t rel_entr_point = 0;
    uint8_t *byte_offset = bytes;

    context->state_flag = STATE_FLAG_NONE;

    while(context->state_flag == STATE_FLAG_NONE) {
        uint32_t remaining_bytes = (uint32_t)(byte_offset + lengths - bytes);

        if (context->rx_state == RX_STATE_DATA) {
            /* Payload of COPY_DATA or COPY_CODE, may arrive in several parts */
            uint32_t chunk_size = context->data_size;
            if (remaining_bytes < chunk_size) chunk_size = remaining_bytes;
            RUNMEM_LOCK();  /* Inputs can be written while tasks are running */
            memcpy(context->data_dest, bytes, chunk_size);
            RUNMEM_UNLOCK();
            context->data_dest += chunk_size;
            context->data_size -= chunk_size;
            bytes += chunk_size;
            if (context->data_size) break;
            context->rx_state = RX_STATE_IDLE;
            continue;
        }

        /* A command is parsed only if it is available with all its arguments */
        if (remaining_bytes < 4) break;
        command = read_u32(bytes);  // Not consumed until the command is complete
        if (remaining_bytes < 4 + command_header_size(command)) break;
        bytes += 4;

        switch(command) {
            case ALLOCATE_DATA:
                size = next_u32(&bytes);
                context->data_memory = allocate_data_memory(size);
                context->data_memory_len = size;
                LOG("ALLOCATE_DATA size=%i mem_addr=%p\n", size, (void*)context->data_memory);
                if (!update_data_offs(context)) context->state_flag = STATE_FLAG_MEM_DIST;
                break;

            case COPY_DATA:
                offs = next_u32(&bytes);
                size = next_u32(&bytes);
                LOG("COPY_DATA offs=%i size=%i\n", offs, size);
                context->data_dest = context->data_memory + offs;
                context->data_size = size;
                context->rx_state = RX_STATE_DATA;
                break;

            case ALLOCATE_CODE:
                size = next_u32(&bytes);
                context->executable_memory = allocate_executable_memory(size);
                context->executable_memory_len = size;
                LOG("ALLOCATE_CODE size=%i mem_addr=%p\n", size, (void*)context->executable_memory);
                if (!update_data_offs(context)) context->state_flag = STATE_FLAG_MEM_DIST;
                break;

            case COPY_CODE:
                offs = next_u32(&bytes);
                size = next_u32(&bytes);
                LOG("COPY_CODE offs=%i size=%i\n", offs, size);
                context->data_dest = context->executable_memory + offs;
                context->data_size = size;
                context->rx_state = RX_STATE_DATA;
                break;

            case PATCH:
                offs = next_u32(&bytes);
                value = (int32_t)next_u32(&bytes);
                patch_mask = next_u32(&bytes);
                encoding = bytes[0];
                shift = bytes[1];
                flags = bytes[2];
                bytes += 4;
                if (!apply_patch(context, offs, value, patch_mask, encoding, shift, flags)) context->state_flag = STATE_FLAG_PATCH;
                break;

            case ENTRY_POINT:
                rel_entr_point = next_u32(&bytes);
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
                offs = next_u32(&bytes);
                size = next_u32(&bytes);
                BLOG("READ_DATA offs=%i size=%i data=", offs, size);
                for (uint32_t i = 0; i < size; i++) {
                    printf("%02X ", context->data_memory[offs + i]);
                }
                printf("\n");
                break;

            case TASK_TABLE:
                offs = next_u32(&bytes);
                size = next_u32(&bytes);
                LOG("TASK_TABLE offs=%i count=%i\n", offs, size);
                runmem_stop(context);
                context->tasks = (task_t *)(context->data_memory + offs);
                context->task_count = size;
                context->tasks_published = 0;
                break;

            case RUN_TASK:
                offs = next_u32(&bytes);
                LOG("RUN_TASK index=%i\n", offs);
                if (offs >= context->task_count || (context->running && context->tasks[offs].period)) {
                    /* Periodic tasks are only run by RUN_TASK if the tasks are stopped */
                    context->state_flag = STATE_FLAG_TASK;
                    break;
                }
                {
                    task_t *task = &context->tasks[offs];
                    task->release_time = runmem_timestamp();
                    read_inputs(context, task);
                    run_copies(context, task->latch_offs, task->latch_count);
                    int ret = measured_job(context, task);
                    run_copies(context, task->publish_offs, task->publish_count);
                    write_outputs(context, task);
                    BLOG("Return value: %i\n", ret);
                }
                break;

            case STEP_TASKS:
                size = next_u32(&bytes);
                LOG("STEP_TASKS ticks=%i\n", size);
                if (context->running) {
                    context->state_flag = STATE_FLAG_TASK;
                    break;
                }
                for (uint32_t i = 0; i < size; i++) {
                    if (!context->tasks_published) tasks_publish(context);
                    tasks_release(context);
                    context->tasks_published = 0;
                }
                tasks_publish(context);
                context->tasks_published = 1;
                break;

            case START_TASKS:
                size = next_u32(&bytes);
                LOG("START_TASKS tick_us=%i\n", size);
                runmem_stop(context);
                for (uint32_t i = 0; i < context->task_count; i++) {
                    task_t *task = &context->tasks[i];
                    task->countdown = 0;
                    task->pending = 0;
                    task->busy = 0;
                    /* Statistics of the real-time execution */
                    task->runs = task->overruns = 0;
                    task->exec_last = task->exec_max = task->exec_sum_lo = task->exec_sum_hi = 0;
                    task->response_max = task->delay_max = 0;
                    task->degraded = task->degradations = 0;
                }
                if (!context->tasks || !size || !runmem_start(context, size)) context->state_flag = STATE_FLAG_TASK;
                break;

            case STOP_TASKS:
                LOG("STOP_TASKS\n");
                runmem_stop(context);
                break;

            case READ_MEM:
                offs = next_u32(&bytes);
                size = next_u32(&bytes);
                LOG("READ_MEM offs=%i size=%i\n", offs, size);
                if (!context->data_memory || offs > context->data_memory_len || size > context->data_memory_len - offs) {
                    context->state_flag = STATE_FLAG_TASK;
                    break;
                }
                printf("COPARUN_DATA %u\n", (unsigned)size);
                fflush(stdout);
                while (size) {
                    /* Copied in parts, each consistent with the publish copies */
                    uint8_t chunk[64];
                    uint32_t chunk_size = size < sizeof(chunk) ? size : (uint32_t)sizeof(chunk);
                    RUNMEM_LOCK();
                    memcpy(chunk, context->data_memory + offs, chunk_size);
                    RUNMEM_UNLOCK();
                    runmem_write_data(chunk, chunk_size);
                    offs += chunk_size;
                    size -= chunk_size;
                }
                fflush(stdout);
                break;

            case LIST_IO:
                LOG("LIST_IO\n");
                runmem_io_list();
                break;

            case FREE_MEMORY:
                LOG("FREE_MENORY\n");
                free_memory(context);
                break;

            case DUMP_CODE:
                LOG("DUMP_CODE\n");
                context->state_flag = STATE_FLAG_DUMP_CODE;
                break;

            case END_COM:
                LOG("END_COM\n");
                context->state_flag = STATE_FLAG_END_COM;
                break;

            default:
                LOG("Unknown command\n");
                context->state_flag = STATE_FLAG_UNKNOWN_COMMAND;
                break;
        }
    }
    return (int)(bytes - byte_offset);
}
