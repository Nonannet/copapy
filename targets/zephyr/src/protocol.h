/*
 * file: protocol.h
 * Description: Messages from the runner to the host (backchannel), see
 * docs/source/runner_transfer.md. All output of the runner, including the
 * text written to stdout, is sent in messages (integers little endian):
 *
 *   uint8   sync byte MSG_SYNC (never part of ASCII text)
 *   uint8   message type MSG_*
 *   uint8   length n of the payload (0..255)
 *   uint8   header check: type ^ length ^ 0xFF
 *   n bytes payload
 *   uint16  CRC-16/XMODEM of type, length, header check and payload
 *
 * Messages are only sent by the main thread.
 */

#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <stdint.h>

#define MSG_SYNC        0xCA
#define MSG_MAX_PAYLOAD 255
#define PROTOCOL_VERSION 1

/* Message types and their payload */
#define MSG_READY    0x01  /* uint8 protocol version: runner started */
#define MSG_DONE     0x02  /* int8 state flag of the parser: stream frame processed */
#define MSG_ERROR    0x03  /* uint8 ERR_*: frame not processed */
#define MSG_CONTROL  0x04  /* uint8 control code: control frame executed */
#define MSG_DATA     0x05  /* data memory read by READ_MEM (part) */
#define MSG_TEXT     0x06  /* text output of the runner (part, lines end with '\n') */
#define MSG_IO       0x07  /* uint16 channel, uint8 IO_KIND_*, name: I/O channel (LIST_IO) */

/* Error codes of MSG_ERROR */
#define ERR_TIMEOUT         1  /* frame incomplete, can be sent again */
#define ERR_CHECKSUM        2  /* CRC of the frame wrong, can be sent again */
#define ERR_FRAME_TOO_LARGE 3
#define ERR_CONTROL_CODE    4  /* unknown control code */

/* I/O channel kinds of MSG_IO */
#define IO_KIND_GPIO_IN    0
#define IO_KIND_GPIO_OUT   1
#define IO_KIND_PWM_OUT    2
#define IO_KIND_ENCODER_IN 3
#define IO_KIND_ADC_IN     4

/* Sends a message with a payload of at most MSG_MAX_PAYLOAD bytes; pending
   text output is sent before */
void send_message(uint8_t type, const void *payload, uint32_t len);

#endif /* PROTOCOL_H */
