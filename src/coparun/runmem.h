/**
 * @file runmem.h
 * @brief Header file for runmem.c, which contains core functions of
 * the runner to receive data, code and patch instructions, perform
 * patching, and jump to the entry point of the copapy program.
 */

#ifndef RUNMEM_H
#define RUNMEM_H

#include <stdint.h>

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

/* Command opcodes used by the parser */
#define ALLOCATE_DATA     1
#define COPY_DATA         2
#define ALLOCATE_CODE     3
#define COPY_CODE         4
#define PATCH             0x1000
#define ENTRY_POINT       7
#define TASK_TABLE        8
#define RUN_PROG         64
#define READ_DATA        65
#define RUN_TASK         66
#define STEP_TASKS       67
#define START_TASKS      68
#define STOP_TASKS       69
#define READ_MEM         70
#define END_COM         256
#define FREE_MEMORY     257
#define DUMP_CODE       258

/* PATCH command arguments:
 *   uint32 offs      address of the patched instruction relative to code memory
 *   int32  value     target address (S + A) relative to code or data memory
 *   uint32 mask      bit mask of the instruction field (PATCH_ENC_BITFIELD only)
 *   uint8  encoding  PATCH_ENC_*
 *   uint8  shift     right shift applied to the calculated value
 *   uint8  flags     PATCH_FLAG_*
 *   uint8  reserved
 *
 * The runner calculates the value to insert as:
 *   S = (flags & DATA ? data_memory : executable_memory) + value
 *   P = executable_memory + offs
 *   with PAGE: S and P rounded down to 4 KiB pages
 *   result = ((flags & PC_REL) ? S - P : S) >> shift
 */

/* Patch encodings: how the result is inserted into the instruction */
#define PATCH_ENC_BITFIELD        0  /* 32 bit word, result placed at the lowest set bit of mask */
#define PATCH_ENC_AARCH64_ADRP    1  /* AArch64 ADRP immhi:immlo (21 bit) */
#define PATCH_ENC_ARM_MOVW_MOVT   2  /* ARM MOVW/MOVT (A1) imm4:imm12 (16 bit) */
#define PATCH_ENC_THUMB_MOVW_MOVT 3  /* Thumb MOVW/MOVT (T3/T1) imm4:i:imm3:imm8 (16 bit) */
#define PATCH_ENC_THUMB_BRANCH    4  /* Thumb B.W/BL (T4/T1) S:J1:J2:imm10:imm11 (24 bit) */
#define PATCH_ENC_RISCV_S_TYPE    5  /* RISC-V store imm[11:5|4:0] (12 bit) */
#define PATCH_ENC_RISCV_B_TYPE    6  /* RISC-V branch imm[12|10:5|4:1|11] (13 bit) */
#define PATCH_ENC_RISCV_J_TYPE    7  /* RISC-V jal imm[20|10:1|11|19:12] (21 bit) */
#define PATCH_ENC_RISCV_CB_TYPE   8  /* RISC-V compressed branch imm[8|4:3|7:6|2:1|5] (16 bit instr.) */
#define PATCH_ENC_RISCV_CJ_TYPE   9  /* RISC-V compressed jump imm[11|4|9:8|10|6|7|3:1|5] (16 bit instr.) */

/* Patch flags: how the result is calculated */
#define PATCH_FLAG_DATA   0x01  /* value is relative to data memory (else code memory) */
#define PATCH_FLAG_PC_REL 0x02  /* subtract address of the patched instruction */
#define PATCH_FLAG_PAGE   0x04  /* round target and instruction address down to 4 KiB pages */

/* Parser states: a command stream can be passed in parts to parse_commands */
#define RX_STATE_IDLE 0  /* next bytes are a command */
#define RX_STATE_DATA 1  /* next bytes are payload of COPY_DATA or COPY_CODE */

/* State flags: reason why parse_commands stopped parsing */
#define STATE_FLAG_NONE             0  /* more bytes of the command stream required */
#define STATE_FLAG_END_COM          1  /* END_COM reached */
#define STATE_FLAG_DUMP_CODE        2  /* DUMP_CODE reached */
#define STATE_FLAG_UNKNOWN_COMMAND -1
#define STATE_FLAG_MEM_DIST        -4  /* code and data memory to far apart */
#define STATE_FLAG_PATCH           -5  /* unknown patch encoding */
#define STATE_FLAG_TASK            -6  /* invalid task index or no task table */

/* Tasks (Logical Execution Time): a periodic task is released every period
 * ticks, starting at tick 0. At its release the latch copies of the task are
 * executed (inputs into the task memory) and the task is run. Its outputs are
 * published (publish copies) at the end of its period, i.e. at its next
 * release. At each tick all publish copies are done before the latch copies.
 *
 * TASK_TABLE arguments:
 *   uint32 offs   offset of the task table (task_t records) in data memory
 *   uint32 count  number of tasks, in order of descending priority
 * RUN_TASK arguments:
 *   uint32 index  runs one job of the task: latch, run, publish
 * STEP_TASKS arguments:
 *   uint32 ticks  simulates the given number of ticks and publishes the
 *                 outputs of the jobs that end at the resulting time
 * START_TASKS arguments:
 *   uint32 tick   tick in microseconds: starts the real-time execution of
 *                 the periodic tasks (if supported by the runner), all tasks
 *                 are released at the start
 * STOP_TASKS: stops the real-time execution after the running jobs
 * READ_MEM arguments:
 *   uint32 offs, uint32 size: writes the line "COPARUN_DATA <size>" followed
 *                 by size bytes of the data memory at offs
 *
 * While the tasks are running, RUN_TASK is only allowed for tasks without
 * period and STEP_TASKS is not allowed.
 */
typedef struct task_s {
    uint32_t entry;          /* offset of the entry function in the code memory */
    uint32_t period;         /* in ticks, 0: task is only run by RUN_TASK */
    int32_t priority;
    uint32_t latch_offs;     /* copy_t records executed at the release */
    uint32_t latch_count;
    uint32_t publish_offs;   /* copy_t records executed at the end of the period */
    uint32_t publish_count;
    uint32_t countdown;      /* ticks until the next release */
    uint32_t pending;        /* job finished, outputs not published yet */
    uint32_t runs;           /* number of jobs run */
    uint32_t overruns;       /* releases skipped because the last job was not finished */
    uint32_t busy;           /* job released and not finished */
} task_t;

/* Copy operation inside the data memory */
typedef struct copy_s {
    uint32_t dst;
    uint32_t src;
    uint32_t size;
} copy_t;

/* Entry point type */
typedef int (*entry_point_t)(void);

#ifdef _WIN64
/* Assembly wrapper to preserve RSI/RDI when calling System V ABI code from Microsoft x64 ABI */
extern int x86_64_abi_shim(entry_point_t entry_point);

static inline int call_entry_point(entry_point_t fp) {
    return x86_64_abi_shim(fp);
}
#else
static inline int call_entry_point(entry_point_t fp) {
    return fp();
}
#endif

/* Struct for run-time memory state */
typedef struct runmem_s {
    uint8_t *data_memory;            // Pointer to data memory
    uint32_t data_memory_len;        // Length of data memory
    uint8_t *executable_memory;      // Pointer to executable memory
    uint32_t executable_memory_len;  // Length of executable memory
    int data_offs;                   // Offset of data memory relative to executable memory
    entry_point_t entr_point;        // Entry point function pointer
    int rx_state;                    // Parser state (RX_STATE_*)
    int state_flag;                  // Result of the last parse_commands call (STATE_FLAG_*)
    uint8_t *data_dest;              // Destination of the remaining payload in RX_STATE_DATA
    uint32_t data_size;              // Size of the remaining payload in RX_STATE_DATA
    task_t *tasks;                   // Task table in data memory (TASK_TABLE)
    uint32_t task_count;             // Number of tasks
    int tasks_published;             // Publish copies of the current tick are done
    volatile int running;            // Real-time execution of the tasks started
} runmem_t;

/* Protection of the copies between task memory and the memory written and
   read by the host against the real-time execution of the tasks */
#ifdef __ZEPHYR__
#include <zephyr/irq.h>
#define RUNMEM_LOCK() unsigned int runmem_lock_key = irq_lock()
#define RUNMEM_UNLOCK() irq_unlock(runmem_lock_key)
#else
#define RUNMEM_LOCK()
#define RUNMEM_UNLOCK()
#endif

/* Functions of the platform (defaults in runmem.c for runners without
   real-time execution): */

/* Starts the real-time execution of the tasks with the given tick, sets
   context->running and returns 1, returns 0 if not supported */
int runmem_start(runmem_t *context, uint32_t tick_us);

/* Stops the real-time execution and waits for the running jobs, clears
   context->running */
void runmem_stop(runmem_t *context);

/* Writes binary data to the output of the runner */
void runmem_write_data(const uint8_t *data, uint32_t size);

/* Initialize the run-time memory state (equal to zero-initialization) */
void runmem_init(runmem_t *context);

/* Command parser: takes the next lengths bytes of the command stream and
   returns the number of bytes processed. The stream can be passed in parts
   of any size: a command with incomplete arguments is not processed, the
   remaining bytes must be passed again together with the following bytes
   (at most 20 bytes). The payload of COPY_DATA and COPY_CODE is processed
   as far as available. Parsing stops with context->state_flag set after
   END_COM, DUMP_CODE or on an error; otherwise state_flag is STATE_FLAG_NONE. */
int parse_commands(runmem_t *context, uint8_t *bytes, uint32_t lengths);

/* Free program and data memory */
void free_memory(runmem_t *context);

/* Publish copies of all periodic tasks whose period ends at the current tick */
void tasks_publish(runmem_t *context);

/* Latch copies of all periodic tasks due at the current tick that are not
   busy and advances to the next tick, returns a bit mask of the released
   tasks (only the first 32 tasks are supported) */
uint32_t tasks_latch(runmem_t *context);

/* Runs the released job of a task */
void task_job(runmem_t *context, uint32_t index);

/* Releases (latch and run) all periodic tasks due at the current tick and
   advances to the next tick */
void tasks_release(runmem_t *context);

#endif /* RUNMEM_H */