/*
 * file: io.c
 * Description: Device I/O channels of the runner. The channels are child
 * nodes of the devicetree node "copapy,io-channels" (see
 * dts/bindings/copapy,io-channels.yaml), numbered in the order of
 * definition. The kind of a channel follows from its property:
 *   gpios        gpio_in / gpio_out  int32, 0: inactive, else active
 *   pwms         pwm_out             float, duty cycle 0..1
 *   encoder      encoder_in          int32, position count
 *   io-channels  adc_in              int32, raw conversion result
 *
 * Channels are read and written in the interrupt of the task tick (see
 * runmem.h). The Zephyr drivers initialize the peripherals, PWM, encoder
 * and ADC are accessed by registers since their driver functions block.
 */

#include <stdio.h>
#include <string.h>
#include <zephyr/kernel.h>
#include <zephyr/init.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/drivers/pwm.h>
#include "runmem.h"

enum kind { GPIO_IN, GPIO_OUT, PWM_OUT, ENCODER_IN, ADC_IN };

static const char *const kind_names[] = {"gpio_in", "gpio_out", "pwm_out", "encoder_in", "adc_in"};

struct channel {
    const char *name;
    enum kind kind;
    struct gpio_dt_spec gpio;    /* GPIO */
    struct pwm_dt_spec pwm;      /* PWM */
    const struct device *dev;    /* encoder, ADC */
    uintptr_t base;              /* registers of the PWM, encoder or ADC */
    uint32_t index;              /* PWM submodule, ADC input */
};

#define IO_NODE DT_COMPAT_GET_ANY_STATUS_OKAY(copapy_io_channels)

#define GPIO_CHANNEL(node) \
    .kind = DT_PROP(node, output) ? GPIO_OUT : GPIO_IN, \
    .gpio = GPIO_DT_SPEC_GET(node, gpios),

#define PWM_CHANNEL(node) \
    .kind = PWM_OUT, \
    .pwm = PWM_DT_SPEC_GET(node), \
    .base = DT_REG_ADDR(DT_PARENT(DT_PWMS_CTLR(node))), \
    .index = DT_PROP(DT_PWMS_CTLR(node), index),

#define ENCODER_CHANNEL(node) \
    .kind = ENCODER_IN, \
    .dev = DEVICE_DT_GET(DT_PHANDLE(node, encoder)), \
    .base = DT_REG_ADDR(DT_PHANDLE(node, encoder)),

#define ADC_CHANNEL(node) \
    .kind = ADC_IN, \
    .dev = DEVICE_DT_GET(DT_IO_CHANNELS_CTLR(node)), \
    .base = DT_REG_ADDR(DT_IO_CHANNELS_CTLR(node)), \
    .index = DT_IO_CHANNELS_INPUT(node),

#define CHANNEL(node) {                                                     \
    .name = DT_NODE_FULL_NAME(node),                                        \
    COND_CODE_1(DT_NODE_HAS_PROP(node, gpios), (GPIO_CHANNEL(node)),        \
    (COND_CODE_1(DT_NODE_HAS_PROP(node, pwms), (PWM_CHANNEL(node)),         \
    (COND_CODE_1(DT_NODE_HAS_PROP(node, encoder), (ENCODER_CHANNEL(node)),  \
    (ADC_CHANNEL(node)))))))                                                \
},

#if DT_NODE_EXISTS(IO_NODE)
static const struct channel channels[] = {DT_FOREACH_CHILD(IO_NODE, CHANNEL)};
#else
static const struct channel channels[] = {};
#endif

#define CHANNEL_COUNT ARRAY_SIZE(channels)

static bool channel_ready[CHANNEL_COUNT + 1];

/* Register access for NXP i.MX RT (FlexPWM, ENC, 12 bit ADC) */
#ifdef CONFIG_HAS_MCUX
#include <fsl_adc.h>
#include <fsl_enc.h>
#include <fsl_pwm.h>

static void pwm_write_pulse(const struct channel *ch, uint32_t pulse) {
    PWM_Type *base = (PWM_Type *)ch->base;

    /* Edge aligned PWM (set up by the driver): output A ends at VAL3, B at VAL5 */
    if (ch->pwm.channel == 0) {
        base->SM[ch->index].VAL3 = (uint16_t)pulse;
    } else {
        base->SM[ch->index].VAL5 = (uint16_t)pulse;
    }
    PWM_SetPwmLdok(base, (uint8_t)(1U << ch->index), true);
}

static int32_t encoder_read(const struct channel *ch) {
    return (int32_t)ENC_GetPositionValue((ENC_Type *)ch->base);
}

static int32_t adc_read(const struct channel *ch) {
    ADC_Type *base = (ADC_Type *)ch->base;
    adc_channel_config_t config = {.channelNumber = ch->index};

    /* Conversion takes about 1-2 us, a missing result is reported as -1 */
    ADC_SetChannelConfig(base, 0, &config);
    for (int i = 0; i < 10000; i++) {
        if (ADC_GetChannelStatusFlags(base, 0)) return (int32_t)ADC_GetChannelConversionValue(base, 0);
    }
    return -1;
}
#else
static void pwm_write_pulse(const struct channel *ch, uint32_t pulse) { ARG_UNUSED(ch); ARG_UNUSED(pulse); }
static int32_t encoder_read(const struct channel *ch) { ARG_UNUSED(ch); return 0; }
static int32_t adc_read(const struct channel *ch) { ARG_UNUSED(ch); return 0; }
#endif

/* PWM period in timer cycles, set at start up */
static uint32_t pwm_period_cycles[CHANNEL_COUNT + 1];

static bool init_channel(uint32_t i) {
    const struct channel *ch = &channels[i];
    uint64_t cycles_per_sec;

    switch (ch->kind) {
        case GPIO_IN:
            return gpio_is_ready_dt(&ch->gpio) && gpio_pin_configure_dt(&ch->gpio, GPIO_INPUT) == 0;
        case GPIO_OUT:
            return gpio_is_ready_dt(&ch->gpio) && gpio_pin_configure_dt(&ch->gpio, GPIO_OUTPUT_INACTIVE) == 0;
        case PWM_OUT:
            if (!pwm_is_ready_dt(&ch->pwm) || pwm_set_pulse_dt(&ch->pwm, 0) != 0 ||
                pwm_get_cycles_per_sec(ch->pwm.dev, ch->pwm.channel, &cycles_per_sec) != 0) return false;
            pwm_period_cycles[i] = (uint32_t)(ch->pwm.period * cycles_per_sec / NSEC_PER_SEC);
            return true;
        default:
            return device_is_ready(ch->dev);
    }
}

static int io_init(void) {
    for (uint32_t i = 0; i < CHANNEL_COUNT; i++) {
        channel_ready[i] = init_channel(i);
    }
    return 0;
}

SYS_INIT(io_init, APPLICATION, CONFIG_APPLICATION_INIT_PRIORITY);

void runmem_io_read(uint32_t channel, uint8_t *value) {
    int32_t result = 0;

    if (channel < CHANNEL_COUNT && channel_ready[channel]) {
        const struct channel *ch = &channels[channel];

        switch (ch->kind) {
            case GPIO_IN:
            case GPIO_OUT:  /* level of the output */
                result = gpio_pin_get_dt(&ch->gpio) > 0;
                break;
            case ENCODER_IN:
                result = encoder_read(ch);
                break;
            case ADC_IN:
                result = adc_read(ch);
                break;
            default:
                break;
        }
    }
    memcpy(value, &result, sizeof(result));
}

void runmem_io_write(uint32_t channel, const uint8_t *value) {
    if (channel >= CHANNEL_COUNT || !channel_ready[channel]) return;
    const struct channel *ch = &channels[channel];

    if (ch->kind == GPIO_OUT) {
        int32_t level;

        memcpy(&level, value, sizeof(level));
        gpio_pin_set_dt(&ch->gpio, level != 0);
    } else if (ch->kind == PWM_OUT) {
        float duty;

        memcpy(&duty, value, sizeof(duty));
        if (!(duty > 0.0f)) duty = 0.0f;  /* also NaN */
        if (duty > 1.0f) duty = 1.0f;
        pwm_write_pulse(ch, (uint32_t)(duty * (float)pwm_period_cycles[channel] + 0.5f));
    }
}

void runmem_io_list(void) {
    for (uint32_t i = 0; i < CHANNEL_COUNT; i++) {
        if (channel_ready[i]) printf("COPARUN_IO %u %s %s\n", (unsigned)i, kind_names[channels[i].kind], channels[i].name);
    }
}
