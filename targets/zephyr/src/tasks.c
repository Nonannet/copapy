/*
 * file: tasks.c
 * Description: Real-time execution of the periodic tasks of a copapy
 * program on Zephyr. A timer with the period of the tick publishes the
 * outputs and latches the inputs of the tasks in its interrupt (see
 * tasks_publish and tasks_latch in runmem.c). The released jobs run on one
 * thread per task with preemptive priorities in the order of the task table
 * (first task: highest priority). The main thread (communication with the
 * host) has a lower priority than all tasks.
 *
 * Load shedding: a supervisor thread with the highest priority measures the
 * CPU load of the task threads in windows of CONFIG_COPARUN_LOAD_WINDOW_MS.
 * If the load exceeds CONFIG_COPARUN_MAX_TASK_LOAD percent, the tasks with
 * the lowest priority are degraded until the load of the remaining tasks is
 * below the limit: they run with lower priority than the main thread, i.e.
 * the host communication is served first and their releases are skipped
 * (overruns) as long as their jobs are not finished. A degraded task is
 * restored (one per window, highest priority first) if its estimated load
 * (mean CPU time per job / period) fits below the limit.
 */

#include <zephyr/kernel.h>
#include "runmem.h"

#define MAX_TASKS CONFIG_COPARUN_MAX_TASKS
#define LOAD_LIMIT CONFIG_COPARUN_MAX_TASK_LOAD  /* percent */
#define LOAD_HYSTERESIS 5                         /* percent */
#define SUPERVISOR_PRIORITY (CONFIG_COPARUN_TASK_PRIORITY - 1)
#define DEGRADED_PRIORITY (CONFIG_MAIN_THREAD_PRIORITY + 1)

BUILD_ASSERT(MAX_TASKS <= 32, "at most 32 tasks are supported by tasks_latch");
BUILD_ASSERT(CONFIG_COPARUN_TASK_PRIORITY >= 1, "the supervisor thread needs a higher priority than the tasks");
BUILD_ASSERT(CONFIG_COPARUN_TASK_PRIORITY + MAX_TASKS <= CONFIG_MAIN_THREAD_PRIORITY,
             "the main thread must have a lower priority than the task threads");
BUILD_ASSERT(DEGRADED_PRIORITY + MAX_TASKS <= K_LOWEST_APPLICATION_THREAD_PRIO,
             "CONFIG_NUM_PREEMPT_PRIORITIES too small for degraded tasks");

K_THREAD_STACK_ARRAY_DEFINE(task_stacks, MAX_TASKS, CONFIG_COPARUN_TASK_STACK_SIZE);
static struct k_thread task_threads[MAX_TASKS];
static struct k_sem task_sems[MAX_TASKS];
static bool threads_created;
static runmem_t *task_context;

/* Load measurement */
static K_SEM_DEFINE(supervisor_sem, 0, 1);
static uint32_t window_ticks;  /* 0: load shedding disabled */
static uint32_t tick_count;
static uint32_t tick_ns;
static uint32_t window_start;
static uint64_t window_cycles[MAX_TASKS];

static void tick_handler(struct k_timer *timer) {
    ARG_UNUSED(timer);

    tasks_publish(task_context);
    uint32_t released = tasks_latch(task_context);

    for (uint32_t i = 0; i < MAX_TASKS; i++) {
        if (released & (1u << i)) k_sem_give(&task_sems[i]);
    }

    if (window_ticks && ++tick_count >= window_ticks) {
        tick_count = 0;
        k_sem_give(&supervisor_sem);
    }
}

static K_TIMER_DEFINE(tick_timer, tick_handler, NULL);

static void task_thread(void *p1, void *p2, void *p3) {
    uint32_t index = (uint32_t)(uintptr_t)p1;

    ARG_UNUSED(p2);
    ARG_UNUSED(p3);

    for (;;) {
        k_sem_take(&task_sems[index], K_FOREVER);
        task_job(task_context, index);
    }
}

static uint64_t thread_cycles(uint32_t index) {
    k_thread_runtime_stats_t stats;

    k_thread_runtime_stats_get(&task_threads[index], &stats);
    return stats.execution_cycles;
}

static void set_degraded(task_t *task, uint32_t index, bool degraded) {
    task->degraded = degraded;
    if (degraded) task->degradations++;
    k_thread_priority_set(&task_threads[index], (degraded ? DEGRADED_PRIORITY : CONFIG_COPARUN_TASK_PRIORITY) + (int)index);
}

/* Estimated load of a task in percent: mean CPU time of its jobs per period */
static uint32_t estimated_load(const task_t *task) {
    uint64_t sum = (uint64_t)task->exec_sum_hi << 32 | task->exec_sum_lo;

    if (!task->runs) return 0;
    return (uint32_t)(sum * 100 / task->runs / ((uint64_t)task->period * tick_ns));
}

static void supervisor_thread(void *p1, void *p2, void *p3) {
    ARG_UNUSED(p1);
    ARG_UNUSED(p2);
    ARG_UNUSED(p3);

    for (;;) {
        k_sem_take(&supervisor_sem, K_FOREVER);

        runmem_t *context = task_context;
        if (!context->running) continue;

        uint32_t now = k_cycle_get_32();
        uint64_t window = now - window_start;
        uint32_t count = MIN(context->task_count, MAX_TASKS);
        uint32_t loads[MAX_TASKS];
        uint32_t load = 0;  /* percent, tasks that are not degraded */

        window_start = now;
        for (uint32_t i = 0; i < count; i++) {
            uint64_t cycles = thread_cycles(i);

            loads[i] = window ? (uint32_t)((cycles - window_cycles[i]) * 100 / window) : 0;
            window_cycles[i] = cycles;
            if (!context->tasks[i].degraded) load += loads[i];
        }

        if (load > LOAD_LIMIT) {
            /* Degrade tasks, lowest priority first, until the load is below the limit */
            for (uint32_t i = count; i-- > 0 && load > LOAD_LIMIT;) {
                task_t *task = &context->tasks[i];

                if (!task->period || task->degraded) continue;
                set_degraded(task, i, true);
                load -= MIN(load, loads[i]);
            }
        } else {
            /* Restore the degraded task with the highest priority if it fits */
            for (uint32_t i = 0; i < count; i++) {
                task_t *task = &context->tasks[i];

                if (!task->degraded) continue;
                if (load + estimated_load(task) + LOAD_HYSTERESIS <= LOAD_LIMIT) set_degraded(task, i, false);
                break;
            }
        }
    }
}

K_THREAD_STACK_DEFINE(supervisor_stack, 1024);
static struct k_thread supervisor;

int runmem_start(runmem_t *context, uint32_t tick_us) {
    if (context->task_count > MAX_TASKS) return 0;

    if (!threads_created) {
        for (uint32_t i = 0; i < MAX_TASKS; i++) {
            k_sem_init(&task_sems[i], 0, 1);
            k_thread_create(&task_threads[i], task_stacks[i], K_THREAD_STACK_SIZEOF(task_stacks[i]),
                            task_thread, (void *)(uintptr_t)i, NULL, NULL,
                            CONFIG_COPARUN_TASK_PRIORITY + (int)i, K_FP_REGS, K_NO_WAIT);
        }
        k_thread_create(&supervisor, supervisor_stack, K_THREAD_STACK_SIZEOF(supervisor_stack),
                        supervisor_thread, NULL, NULL, NULL, SUPERVISOR_PRIORITY, 0, K_NO_WAIT);
        threads_created = true;
    }

    task_context = context;
    for (uint32_t i = 0; i < context->task_count; i++) {
        set_degraded(&context->tasks[i], i, false);
        window_cycles[i] = thread_cycles(i);
    }
    tick_ns = tick_us * 1000u;
    tick_count = 0;
    window_ticks = LOAD_LIMIT < 100 ? CONFIG_COPARUN_LOAD_WINDOW_MS * 1000u / tick_us : 0;
    window_start = k_cycle_get_32();

    context->running = 1;
    k_timer_start(&tick_timer, K_NO_WAIT, K_USEC(tick_us));
    return 1;
}

/* Execution time measurement: hardware cycle counter of the system clock
   (CPU clock on Cortex-M with SysTick), CPU time of the threads by the
   runtime statistics of the kernel (time of interrupts during a job is
   counted for the job) */
uint32_t runmem_timestamp(void) {
    return k_cycle_get_32();
}

uint32_t runmem_cpu_time(void) {
    k_thread_runtime_stats_t stats;

    k_thread_runtime_stats_get(k_current_get(), &stats);
    return (uint32_t)stats.execution_cycles;
}

uint32_t runmem_to_ns(uint32_t delta) {
    return (uint32_t)k_cyc_to_ns_floor64(delta);
}

void runmem_stop(runmem_t *context) {
    if (!context->running) return;

    k_timer_stop(&tick_timer);
    context->running = 0;

    /* Wait for the released jobs (degraded tasks run while the main thread sleeps) */
    for (uint32_t i = 0; i < context->task_count; i++) {
        while (context->tasks[i].busy) k_msleep(1);
    }
}
