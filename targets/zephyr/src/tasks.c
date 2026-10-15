/*
 * file: tasks.c
 * Description: Real-time execution of the periodic tasks of a copapy
 * program on Zephyr. A timer with the period of the tick publishes the
 * outputs and latches the inputs of the tasks in its interrupt (see
 * tasks_publish and tasks_latch in runmem.c). The released jobs run on one
 * thread per task with preemptive priorities in the order of the task table
 * (first task: highest priority). The main thread (communication with the
 * host) has a lower priority than all tasks.
 */

#include <zephyr/kernel.h>
#include "runmem.h"

#define MAX_TASKS CONFIG_COPARUN_MAX_TASKS

BUILD_ASSERT(MAX_TASKS <= 32, "at most 32 tasks are supported by tasks_latch");
BUILD_ASSERT(CONFIG_COPARUN_TASK_PRIORITY + MAX_TASKS <= CONFIG_MAIN_THREAD_PRIORITY,
             "the main thread must have a lower priority than the task threads");

K_THREAD_STACK_ARRAY_DEFINE(task_stacks, MAX_TASKS, CONFIG_COPARUN_TASK_STACK_SIZE);
static struct k_thread task_threads[MAX_TASKS];
static struct k_sem task_sems[MAX_TASKS];
static bool threads_created;
static runmem_t *task_context;

static void tick_handler(struct k_timer *timer) {
    ARG_UNUSED(timer);

    tasks_publish(task_context);
    uint32_t released = tasks_latch(task_context);

    for (uint32_t i = 0; i < MAX_TASKS; i++) {
        if (released & (1u << i)) k_sem_give(&task_sems[i]);
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

int runmem_start(runmem_t *context, uint32_t tick_us) {
    if (context->task_count > MAX_TASKS) return 0;

    if (!threads_created) {
        for (uint32_t i = 0; i < MAX_TASKS; i++) {
            k_sem_init(&task_sems[i], 0, 1);
            k_thread_create(&task_threads[i], task_stacks[i], K_THREAD_STACK_SIZEOF(task_stacks[i]),
                            task_thread, (void *)(uintptr_t)i, NULL, NULL,
                            CONFIG_COPARUN_TASK_PRIORITY + (int)i, K_FP_REGS, K_NO_WAIT);
        }
        threads_created = true;
    }

    task_context = context;
    context->running = 1;
    k_timer_start(&tick_timer, K_NO_WAIT, K_USEC(tick_us));
    return 1;
}

void runmem_stop(runmem_t *context) {
    if (!context->running) return;

    k_timer_stop(&tick_timer);
    context->running = 0;

    /* Wait for the released jobs */
    for (uint32_t i = 0; i < context->task_count; i++) {
        while (context->tasks[i].busy) k_msleep(1);
    }
}
