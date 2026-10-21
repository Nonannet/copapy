/*
 * file: main.c
 * Description: Runner for Zephyr targets. Command streams are received
 * framed over a UART, the output of the runner is sent in binary messages
 * (protocol.h) over the same UART. The UART is selected by the chosen
 * devicetree node "coparun,uart" (e.g. a USB CDC ACM device), by default it
 * is the console. The protocol is described in docs/source/runner_transfer.md.
 *
 * Stream frame (integers little endian):
 *   4 bytes  magic "CPY1"
 *   uint32   length n of the following part of the command stream in bytes,
 *            at most CONFIG_COPARUN_MAX_FRAME_SIZE
 *   n bytes  part of the command stream
 *   uint32   CRC-32 (IEEE 802.3, as zlib.crc32) of the n bytes
 *
 * A command stream can be split at any position into several frames. Each
 * frame is answered with the output of the runner followed by a MSG_DONE
 * message with the state flag of the parser (0: command stream is continued
 * by the next frame, 1: END_COM reached, 2: DUMP_CODE reached, negative:
 * error), or by MSG_ERROR if the frame was not valid. A frame is processed
 * only after its checksum was verified, a frame answered with ERR_TIMEOUT or
 * ERR_CHECKSUM can be sent again. The next frame must not be sent before the
 * answer was received. The run-time memory state is kept between frames and
 * command streams.
 *
 * Control frame:
 *   4 bytes  magic "CPYC"
 *   uint32   0: abort the current command stream, the next frame starts a
 *               new one
 *            1: reset the device
 *            2: enter the bootloader, Teensy only
 *            answered by MSG_CONTROL with the code before it is executed
 *
 * Bytes in front of a magic are ignored. After start up MSG_READY is sent.
 * Text written to stdout is sent in MSG_TEXT messages, the data of READ_MEM
 * in MSG_DATA messages.
 *
 * Periodic tasks are run in real time after START_TASKS (see tasks.c), the
 * communication with the host continues meanwhile.
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/sys/byteorder.h>
#include <zephyr/sys/crc.h>
#include <zephyr/sys/libc-hooks.h>
#include <zephyr/sys/printk.h>
#include <zephyr/sys/reboot.h>
#include <zephyr/sys/ring_buffer.h>
#include "runmem.h"
#include "tasks.h"
#include "protocol.h"

#define RX_TIMEOUT K_MSEC(CONFIG_COPARUN_RX_TIMEOUT_MS)

/* UART of the runner: chosen node "coparun,uart", by default the console */
#if DT_HAS_CHOSEN(coparun_uart)
#define COPARUN_UART_NODE DT_CHOSEN(coparun_uart)
#else
#define COPARUN_UART_NODE DT_CHOSEN(zephyr_console)
#endif

static const struct device *const uart_dev = DEVICE_DT_GET(COPARUN_UART_NODE);

static const uint8_t frame_magic[3] = {'C', 'P', 'Y'};
#define FRAME_TYPE_STREAM  '1'
#define FRAME_TYPE_CONTROL 'C'

#define CONTROL_ABORT      0
#define CONTROL_RESET      1
#define CONTROL_BOOTLOADER 2

RING_BUF_DECLARE(rx_ring, CONFIG_COPARUN_RX_RING_SIZE);
static K_SEM_DEFINE(rx_sem, 0, 1);
static volatile bool rx_paused;

RING_BUF_DECLARE(tx_ring, CONFIG_COPARUN_TX_RING_SIZE);
static K_SEM_DEFINE(tx_sem, 0, 1);
static volatile bool tx_stalled;

/* Command stream of a frame, preceded by the bytes of an incomplete command
   of the previous frame (at most 19 bytes) */
#define MAX_PENDING 20
static uint8_t frame_buffer[MAX_PENDING + CONFIG_COPARUN_MAX_FRAME_SIZE];
static uint32_t pending;

/* Text output, sent at the end of a line, if the buffer is full or before
   the next message */
static uint8_t text_buffer[MSG_MAX_PAYLOAD];
static uint32_t text_len;

static runmem_t targ;

static void uart_isr(const struct device *dev, void *user_data) {
    ARG_UNUSED(user_data);

    while (uart_irq_update(dev) && uart_irq_is_pending(dev)) {
        if (uart_irq_rx_ready(dev)) {
            uint8_t *data;
            uint32_t space = ring_buf_put_claim(&rx_ring, &data, CONFIG_COPARUN_RX_RING_SIZE);

            if (space) {
                int len = uart_fifo_read(dev, data, (int)space);

                ring_buf_put_finish(&rx_ring, len > 0 ? (uint32_t)len : 0);
            } else {
                /* Receiving is paused until the main thread has read data:
                   back pressure to the host if the driver supports it (USB) */
                uart_irq_rx_disable(dev);
                rx_paused = true;
            }
            k_sem_give(&rx_sem);
        }
        if (uart_irq_tx_ready(dev)) {
            uint8_t *data;
            uint32_t len = ring_buf_get_claim(&tx_ring, &data, CONFIG_COPARUN_TX_RING_SIZE);

            if (len) {
                int sent = uart_fifo_fill(dev, data, (int)len);

                ring_buf_get_finish(&tx_ring, sent > 0 ? (uint32_t)sent : 0);
                tx_stalled = false;
            } else {
                uart_irq_tx_disable(dev);
            }
            k_sem_give(&tx_sem);
        }
    }
}

/* Output: written without loss as long as the host reads it. The poll mode
   of a USB CDC ACM device drops characters if its buffer is full. */
static void tx_put(const uint8_t *data, uint32_t len) {
    while (len) {
        uint32_t n = ring_buf_put(&tx_ring, data, len);

        data += n;
        len -= n;
        if (!len) break;
        /* Output is dropped if the host does not read it */
        if (tx_stalled) return;
        uart_irq_tx_enable(uart_dev);
        if (k_sem_take(&tx_sem, K_MSEC(CONFIG_COPARUN_TX_TIMEOUT_MS)) != 0) {
            tx_stalled = true;
            return;
        }
    }
}

static void put_message(uint8_t type, const void *payload, uint32_t len) {
    uint8_t header[4] = {MSG_SYNC, type, (uint8_t)len, (uint8_t)(type ^ len ^ 0xFF)};
    uint8_t crc[2];

    sys_put_le16(crc16_itu_t(crc16_itu_t(0, header + 1, 3), payload, len), crc);
    tx_put(header, sizeof(header));
    tx_put(payload, len);
    tx_put(crc, sizeof(crc));
    uart_irq_tx_enable(uart_dev);
}

static void text_flush(void) {
    if (text_len) {
        put_message(MSG_TEXT, text_buffer, text_len);
        text_len = 0;
    }
}

void send_message(uint8_t type, const void *payload, uint32_t len) {
    text_flush();
    put_message(type, payload, len);
}

static void send_byte(uint8_t type, uint8_t value) {
    send_message(type, &value, 1);
}

static int stdout_write(int c) {
    text_buffer[text_len++] = (uint8_t)c;
    if (c == '\n' || text_len == sizeof(text_buffer)) text_flush();
    return c;
}

void runmem_write_data(const uint8_t *data, uint32_t size) {
    while (size) {
        uint32_t len = size < MSG_MAX_PAYLOAD ? size : MSG_MAX_PAYLOAD;

        send_message(MSG_DATA, data, len);
        data += len;
        size -= len;
    }
}

/* Reads len bytes, returns 0 if no data was received for the time of timeout */
static int read_bytes(uint8_t *dest, uint32_t len, k_timeout_t timeout) {
    while (len) {
        uint32_t n = ring_buf_get(&rx_ring, dest, len);

        dest += n;
        len -= n;
        if (rx_paused) {
            /* Space in the ring buffer again */
            rx_paused = false;
            uart_irq_rx_enable(uart_dev);
        }
        if (!n && k_sem_take(&rx_sem, timeout) != 0) return 0;
    }
    return 1;
}

/* Waits for the magic of a frame, returns its last character (frame type) */
static uint8_t wait_for_magic(void) {
    uint32_t matched = 0;
    uint8_t c;

    for (;;) {
        read_bytes(&c, 1, K_FOREVER);
        if (matched == sizeof(frame_magic)) {
            if (c == FRAME_TYPE_STREAM || c == FRAME_TYPE_CONTROL) return c;
            matched = 0;
        }
        if (c == frame_magic[matched]) {
            matched++;
        } else {
            matched = (c == frame_magic[0]);
        }
    }
}

/* Receives the remaining stream frame after the magic and appends the
   command stream to frame_buffer, returns ERR_* on failure, else 0 */
static uint8_t receive_frame(uint32_t *len) {
    uint8_t word[4];

    if (!read_bytes(word, sizeof(word), RX_TIMEOUT)) return ERR_TIMEOUT;
    *len = sys_get_le32(word);

    if (*len > CONFIG_COPARUN_MAX_FRAME_SIZE) return ERR_FRAME_TOO_LARGE;
    if (!read_bytes(frame_buffer + pending, *len, RX_TIMEOUT) ||
        !read_bytes(word, sizeof(word), RX_TIMEOUT)) return ERR_TIMEOUT;
    if (sys_get_le32(word) != crc32_ieee(frame_buffer + pending, *len)) return ERR_CHECKSUM;
    return 0;
}

static void process_stream_frame(void) {
    uint32_t len;
    uint8_t error = receive_frame(&len);

    if (error) {
        send_byte(MSG_ERROR, error);
        return;
    }

    uint32_t consumed = (uint32_t)parse_commands(&targ, frame_buffer, pending + len);

    if (targ.state_flag == STATE_FLAG_NONE) {
        /* Keep the bytes of an incomplete command for the next frame */
        pending += len - consumed;
        memmove(frame_buffer, frame_buffer + consumed, pending);
    } else {
        /* End of the command stream or parser out of sync */
        pending = 0;
        targ.rx_state = RX_STATE_IDLE;
    }
    fflush(stdout);
    send_byte(MSG_DONE, (uint8_t)(int8_t)targ.state_flag);
}

static void process_control_frame(void) {
    uint8_t word[4];

    if (!read_bytes(word, sizeof(word), RX_TIMEOUT)) {
        send_byte(MSG_ERROR, ERR_TIMEOUT);
        return;
    }

    switch (sys_get_le32(word)) {
        case CONTROL_ABORT:
            pending = 0;
            targ.rx_state = RX_STATE_IDLE;
            send_byte(MSG_CONTROL, CONTROL_ABORT);
            break;

        case CONTROL_RESET:
            send_byte(MSG_CONTROL, CONTROL_RESET);
            k_msleep(100);  /* Let the output drain */
            sys_reboot(SYS_REBOOT_COLD);
            break;

#ifdef CONFIG_BOARD_TEENSY41
        case CONTROL_BOOTLOADER:
            send_byte(MSG_CONTROL, CONTROL_BOOTLOADER);
            k_msleep(100);
            /* Request of the bootloader chip of the Teensy (HalfKay) */
            __asm__ volatile("bkpt #251");
            break;
#endif

        default:
            send_byte(MSG_ERROR, ERR_CONTROL_CODE);
            break;
    }
}

int main(void) {
    if (!device_is_ready(uart_dev)) {
        printk("coparun: UART not ready\n");
        return 0;
    }

    uart_irq_callback_set(uart_dev, uart_isr);
    uart_irq_rx_enable(uart_dev);
    __stdout_hook_install(stdout_write);

    send_byte(MSG_READY, PROTOCOL_VERSION);

    for (;;) {
        if (wait_for_magic() == FRAME_TYPE_STREAM) {
            process_stream_frame();
        } else {
            process_control_frame();
        }
    }
    return 0;
}
