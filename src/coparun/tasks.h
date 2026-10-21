/**
 * @file tasks.h
 * @brief Header file for tasks.c: periodic tasks of a copapy program and
 * reading of the data memory by the host (commands TASK_TABLE, RUN_TASK,
 * STEP_TASKS, START_TASKS, STOP_TASKS and READ_MEM of runmem.h).
 */

#ifndef TASKS_H
#define TASKS_H

#include <stdint.h>
#include "runmem.h"

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
 *                 are released at the start. The statistics are reset.
 * STOP_TASKS: stops the real-time execution after the running jobs
 * READ_MEM arguments:
 *   uint32 offs, uint32 size: sends size bytes of the data memory at offs
 *                 in parts of at most 252 bytes (runmem_write_data)
 *
 * LIST_IO: sends the list of the I/O channels of the device (runmem_io_list)
 *
 * While the tasks are running, RUN_TASK is only allowed for tasks without
 * period and STEP_TASKS is not allowed.
 *
 * Device I/O: input channels are read into the data memory at the release
 * of a task before its latch copies (io_in records), output channels are
 * written from the data memory after its publish copies (io_out records).
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
    uint32_t release_time;   /* time stamp of the last release (runmem_timestamp) */
    /* Statistics of the jobs in ns, reset by START_TASKS (with runs and overruns) */
    uint32_t exec_last;      /* CPU time of the last job (without preemption) */
    uint32_t exec_max;       /* maximum CPU time of a job */
    uint32_t exec_sum_lo;    /* sum of the CPU times of all jobs (64 bit) */
    uint32_t exec_sum_hi;
    uint32_t response_max;   /* maximum time from release to the end of a job */
    uint32_t delay_max;      /* maximum time from release to the start of a job */
    /* Load shedding of the real-time execution (if supported by the runner) */
    uint32_t degraded;       /* task runs with lower priority than the host communication */
    uint32_t degradations;   /* number of times the task was degraded */
    uint32_t io_in_offs;     /* io_t records of input channels, read at the release */
    uint32_t io_in_count;
    uint32_t io_out_offs;    /* io_t records of output channels, written at the publish */
    uint32_t io_out_count;
} task_t;

/* Device I/O channel and its value (4 bytes) in the data memory */
typedef struct io_s {
    uint32_t channel;
    uint32_t addr;
} io_t;

/* Copy operation inside the data memory */
typedef struct copy_s {
    uint32_t dst;
    uint32_t src;
    uint32_t size;
} copy_t;

/* Functions of the platform (defaults in tasks.c for runners without
   real-time execution and device I/O): */

/* Starts the real-time execution of the tasks with the given tick, sets
   context->running and returns 1, returns 0 if not supported */
int runmem_start(runmem_t *context, uint32_t tick_us);

/* Stops the real-time execution and waits for the running jobs, clears
   context->running */
void runmem_stop(runmem_t *context);

/* Sends a part of the data read by READ_MEM to the host (binary) */
void runmem_write_data(const uint8_t *data, uint32_t size);

/* Time stamp counter (wrapping) for the execution time measurement */
uint32_t runmem_timestamp(void);

/* CPU time counter of the current thread (wrapping, same unit as the time
   stamp counter) */
uint32_t runmem_cpu_time(void);

/* Converts a difference of counter values to ns */
uint32_t runmem_to_ns(uint32_t delta);

/* Device I/O channels (ISR safe): read a channel into or write a channel from
   4 bytes of the data memory (GPIO: int32, 0 inactive, else active) */
void runmem_io_read(uint32_t channel, uint8_t *value);
void runmem_io_write(uint32_t channel, const uint8_t *value);

/* Commands of the parser, each returns 0 if the command is not allowed
   (STATE_FLAG_TASK), else 1 */

/* TASK_TABLE: stops the real-time execution and sets the task table */
int tasks_set_table(runmem_t *context, uint32_t offs, uint32_t count);

/* RUN_TASK: runs one job of a task (latch, run, publish) */
int tasks_run(runmem_t *context, uint32_t index);

/* STEP_TASKS: simulates ticks while the real-time execution is stopped */
int tasks_step(runmem_t *context, uint32_t ticks);

/* STOP_TASKS: stops the real-time execution after the running jobs */
void tasks_stop(runmem_t *context);

/* Stops the real-time execution and removes the task table (free_memory) */
void tasks_clear(runmem_t *context);

/* START_TASKS: resets the task states and statistics and starts the
   real-time execution */
int tasks_start(runmem_t *context, uint32_t tick_us);

/* READ_MEM: sends a range of the data memory to the host */
int read_mem(runmem_t *context, uint32_t offs, uint32_t size);

/* Scheduling functions for the real-time execution of a runner: */

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

#endif /* TASKS_H */
