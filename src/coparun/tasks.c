/*
 * file: tasks.c
 * Description: Periodic tasks of a copapy program (Logical Execution Time,
 * see tasks.h) and reading of the data memory by the host. The commands
 * TASK_TABLE, RUN_TASK, STEP_TASKS, START_TASKS, STOP_TASKS and READ_MEM of
 * the parser in runmem.c are implemented here, as well as the scheduling
 * functions used by the real-time execution of a runner (e.g.
 * targets/zephyr/src/tasks.c) and the default platform functions for runners
 * without real-time execution and device I/O.
 */

#include <stdio.h>
#include <string.h>
#include "runmem.h"
#include "tasks.h"

#ifndef __ZEPHYR__
#include <time.h>

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
#endif

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

int tasks_set_table(runmem_t *context, uint32_t offs, uint32_t count) {
    runmem_stop(context);
    context->tasks = (task_t *)(context->data_memory + offs);
    context->task_count = count;
    context->tasks_published = 0;
    return 1;
}

int tasks_run(runmem_t *context, uint32_t index) {
    /* Periodic tasks are only run by RUN_TASK if the tasks are stopped */
    if (index >= context->task_count || (context->running && context->tasks[index].period)) return 0;

    task_t *task = &context->tasks[index];
    task->release_time = runmem_timestamp();
    read_inputs(context, task);
    run_copies(context, task->latch_offs, task->latch_count);
    int ret = measured_job(context, task);
    run_copies(context, task->publish_offs, task->publish_count);
    write_outputs(context, task);
    BLOG("Return value: %i\n", ret);
    (void)ret;
    return 1;
}

int tasks_step(runmem_t *context, uint32_t ticks) {
    if (context->running) return 0;
    for (uint32_t i = 0; i < ticks; i++) {
        if (!context->tasks_published) tasks_publish(context);
        tasks_release(context);
        context->tasks_published = 0;
    }
    tasks_publish(context);
    context->tasks_published = 1;
    return 1;
}

void tasks_stop(runmem_t *context) {
    runmem_stop(context);
}

void tasks_clear(runmem_t *context) {
    runmem_stop(context);
    context->tasks = NULL;
    context->task_count = 0;
}

int tasks_start(runmem_t *context, uint32_t tick_us) {
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
    return context->tasks && tick_us && runmem_start(context, tick_us);
}

int read_mem(runmem_t *context, uint32_t offs, uint32_t size) {
    if (!context->data_memory || offs > context->data_memory_len || size > context->data_memory_len - offs) return 0;

    fflush(stdout);
    while (size) {
        /* Sent in parts (one message each on the serial runners), copied in
           pieces of 64 bytes, each consistent with the publish copies */
        uint8_t chunk[252];
        uint32_t chunk_size = size < sizeof(chunk) ? size : (uint32_t)sizeof(chunk);
        for (uint32_t i = 0; i < chunk_size; i += 64) {
            uint32_t n = chunk_size - i < 64 ? chunk_size - i : 64;
            RUNMEM_LOCK();
            memcpy(chunk + i, context->data_memory + offs + i, n);
            RUNMEM_UNLOCK();
        }
        runmem_write_data(chunk, chunk_size);
        offs += chunk_size;
        size -= chunk_size;
    }
    fflush(stdout);
    return 1;
}
