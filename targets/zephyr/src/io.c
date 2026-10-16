/*
 * file: io.c
 * Description: Device I/O channels of the runner. The channels are defined
 * in the devicetree (see dts/bindings/copapy,gpio-channels.yaml) and are
 * numbered in the order of definition. Channels are read and written in the
 * interrupt of the task tick (see runmem.h), the drivers must be ISR safe.
 */

#include <stdio.h>
#include <string.h>
#include <zephyr/kernel.h>
#include <zephyr/init.h>
#include <zephyr/drivers/gpio.h>
#include "runmem.h"

struct gpio_channel {
    const char *name;
    struct gpio_dt_spec spec;
    bool output;
};

#define GPIO_CHANNELS_NODE DT_COMPAT_GET_ANY_STATUS_OKAY(copapy_gpio_channels)

#if DT_NODE_EXISTS(GPIO_CHANNELS_NODE)
#define GPIO_CHANNEL(node) \
    {DT_NODE_FULL_NAME(node), GPIO_DT_SPEC_GET(node, gpios), DT_PROP(node, output)},

static const struct gpio_channel gpio_channels[] = {
    DT_FOREACH_CHILD(GPIO_CHANNELS_NODE, GPIO_CHANNEL)
};
#define GPIO_CHANNEL_COUNT ARRAY_SIZE(gpio_channels)
#else
static const struct gpio_channel *const gpio_channels = NULL;
#define GPIO_CHANNEL_COUNT 0
#endif

/* Channel configuration at start up: outputs inactive */
static bool channel_ready[GPIO_CHANNEL_COUNT + 1];

static int io_init(void) {
    for (uint32_t i = 0; i < GPIO_CHANNEL_COUNT; i++) {
        const struct gpio_channel *ch = &gpio_channels[i];

        channel_ready[i] = gpio_is_ready_dt(&ch->spec) &&
            gpio_pin_configure_dt(&ch->spec, ch->output ? GPIO_OUTPUT_INACTIVE : GPIO_INPUT) == 0;
    }
    return 0;
}

SYS_INIT(io_init, APPLICATION, CONFIG_APPLICATION_INIT_PRIORITY);

void runmem_io_read(uint32_t channel, uint8_t *value) {
    int32_t level = 0;

    if (channel < GPIO_CHANNEL_COUNT && channel_ready[channel]) {
        level = gpio_pin_get_dt(&gpio_channels[channel].spec) > 0;
    }
    memcpy(value, &level, sizeof(level));
}

void runmem_io_write(uint32_t channel, const uint8_t *value) {
    int32_t level;

    if (channel >= GPIO_CHANNEL_COUNT || !channel_ready[channel] || !gpio_channels[channel].output) return;
    memcpy(&level, value, sizeof(level));
    gpio_pin_set_dt(&gpio_channels[channel].spec, level != 0);
}

void runmem_io_list(void) {
    for (uint32_t i = 0; i < GPIO_CHANNEL_COUNT; i++) {
        if (!channel_ready[i]) continue;
        printf("COPARUN_IO %u %s %s\n", (unsigned)i,
               gpio_channels[i].output ? "gpio_out" : "gpio_in", gpio_channels[i].name);
    }
}
