# Runner transfer protocol

This page describes how copapy (the host) talks to a runner on a device over
a serial byte stream, for example the Zephyr runner (`targets/zephyr`) over a
UART or a USB CDC ACM port. `copapy.SerialTarget` implements the host side.
The runner side is in `targets/zephyr/src/main.c` and
`targets/zephyr/src/protocol.h`.

There are two directions:

- **Host → runner:** command streams, split into checksummed *frames*, plus
  *control frames*.
- **Runner → host (backchannel):** binary *messages*. These carry all output
  of the runner: answers to frames, data read from the device memory, the list
  of I/O channels and the runner's text output.

## Byte order

There are two byte orders:

- **Fixed little endian** for all fields of the transport layer: frame
  length and CRC-32, control code, message CRC-16 and the `uint16` channel
  number of `IO`. These fields are the same on every target.
- **Target byte order** (`sdb.byteorder` of the stencil database, i.e. the
  byte order of the device's CPU) for the content: the command stream inside
  stream frames (command words and arguments, written by
  `data_writer(sdb.byteorder)` and read natively by `parse_commands`) and the
  payload of `DATA`.

`READ_MEM` copies the data memory unchanged into `DATA` messages, so the data
arrives in the target byte order, as the values are stored on the device. The
host decodes it with the target byte order (`read_value`, task statistics).

All other message payloads are single bytes or text and have no byte order.

## Host → runner

### Stream frame

| Offset | Size | Field                                                        |
|--------|------|--------------------------------------------------------------|
| 0      | 4    | Magic `"CPY1"`                                               |
| 4      | 4    | Length *n* of the command stream part (≤ `CONFIG_COPARUN_MAX_FRAME_SIZE`), little endian |
| 8      | *n*  | Part of the command stream, target byte order                |
| 8+*n*  | 4    | CRC-32 (IEEE 802.3, as `zlib.crc32`) of the *n* bytes, little endian |

A command stream (see `src/coparun/runmem.h` for the commands) can be split
anywhere into frames. The runner checks the CRC of a frame before it processes
the frame. Each frame gets exactly one final answer: `DONE` or `ERROR`. The
host does not send the next frame until that answer has arrived.

### Control frame

| Offset | Size | Field                    |
|--------|------|--------------------------|
| 0      | 4    | Magic `"CPYC"`           |
| 4      | 4    | Control code, little endian |

| Code | Action                                                           |
|------|------------------------------------------------------------------|
| 0    | Abort: drop the current command stream; the next frame starts a new one |
| 1    | Reset the device                                                 |
| 2    | Enter the bootloader (Teensy only)                               |

The runner sends `CONTROL` with the code before it carries out the action.
The runner ignores bytes in front of a magic. That lets a new host session
start while the runner is still in the middle of an old one: send an abort and
skip all messages up to its `CONTROL` answer.

## Runner → host (backchannel)

### Message layout

| Offset | Size | Field                                                         |
|--------|------|---------------------------------------------------------------|
| 0      | 1    | Sync byte `0xCA`                                              |
| 1      | 1    | Message type                                                  |
| 2      | 1    | Length *n* of the payload (0…255)                              |
| 3      | 1    | Header check: `type ^ n ^ 0xFF`                               |
| 4      | *n*  | Payload                                                       |
| 4+*n*  | 2    | CRC-16/XMODEM (poly 0x1021, init 0) of bytes 1 … 3+*n*, little endian |

The overhead is 6 bytes per message. The CRC is `crc16_itu_t(0, …)` in Zephyr
and `binascii.crc_hqx(…, 0)` in Python.

### Message types

| Type   | Name      | Payload                                                       | Sent                                          |
|--------|-----------|---------------------------------------------------------------|-----------------------------------------------|
| `0x01` | `READY`   | `uint8` protocol version (currently 1)                        | Once after the runner starts                  |
| `0x02` | `DONE`    | `int8` parser state flag                                      | After each processed stream frame             |
| `0x03` | `ERROR`   | `uint8` error code                                            | Instead of `DONE` if the frame was invalid    |
| `0x04` | `CONTROL` | `uint8` control code                                          | Answer to a control frame                     |
| `0x05` | `DATA`    | 1…255 bytes of data memory, target byte order                 | `READ_MEM`, as many messages as needed        |
| `0x06` | `TEXT`    | 1…255 bytes of text output                                    | Output of the runner (`printf`, logging)      |
| `0x07` | `IO`      | `uint16` channel (little endian), `uint8` kind, name (UTF-8, no terminator) | `LIST_IO`, one message per channel |

`DONE` state flags (`STATE_FLAG_*` in `runmem.h`):

| Value | Meaning                                                           |
|-------|-------------------------------------------------------------------|
| 0     | The command stream continues in the next frame                    |
| 1     | `END_COM` reached                                                 |
| 2     | `DUMP_CODE` reached                                               |
| < 0   | Error: −1 unknown command, −4 code and data memory too far apart, −5 unknown patch encoding, −6 invalid task operation or memory range |

`ERROR` codes:

| Code | Meaning                         | Can the frame be sent again? |
|------|---------------------------------|------------------------------|
| 1    | Timeout, frame incomplete       | Yes                          |
| 2    | CRC of the frame wrong          | Yes                          |
| 3    | Frame too large                 | No                           |
| 4    | Unknown control code            | No                           |

`IO` channel kinds: 0 `gpio_in`, 1 `gpio_out`, 2 `pwm_out`, 3 `encoder_in`,
4 `adc_in`.

### Ordering

The runner answers a stream frame with zero or more `TEXT`, `DATA` and `IO`
messages, then one `DONE` or `ERROR`. Within one command stream, the host
appends the payloads of all `DATA` messages in the order they arrive. A
`READ_MEM` command of *s* bytes produces ⌈*s*/252⌉ `DATA` messages; the core
copies the memory in parts of 252 bytes.

Text written to `stdout` is collected in a buffer of 255 bytes. The runner
sends that buffer as a `TEXT` message at the end of a line, when the buffer is
full, and before any other message, so the order of the output is kept. A
line can span several `TEXT` messages. The host joins them and splits the
result at `\n`. Only the main thread of the runner sends messages.

### Synchronisation and errors

The sync byte `0xCA` never occurs in ASCII text. On a board where the runner
shares its UART with the Zephyr console (for example mps2/an500 in qemu),
console output such as the boot banner arrives between messages. The host
decoder (`copapy._serial.MessageReader`) works like this:

1. It reads bytes until it finds a sync byte. It treats the other bytes as
   text output.
2. It reads the three header bytes. If the header check is wrong, the sync
   byte was not the start of a message, and the decoder continues searching
   at the byte after it. Because of this check, the decoder does not wait for
   the payload of a false header.
3. It reads the payload and the CRC. If the CRC is wrong, it pushes the bytes
   back and continues searching at the byte after the sync byte.

If a message is lost, the host gets no answer to its frame and times out.
`SerialTarget` sends a frame again after `ERROR` with code 1 or 2.

### Efficiency

A `read_value` of a 32-bit value (`READ_MEM` + `END_COM`) is answered with
17 bytes: `DATA` (6 + 4) and `DONE` (6 + 1). For large reads the overhead is
6 bytes per 252 data bytes (2.4 %). The host reads each message with three
`read` calls: sync byte, header and the rest.

### Examples

```
READY, version 1            ca 01 01 ff 01 5a 52
DONE, END_COM reached       ca 02 01 fc 01 d5 9c
DATA, float 1.5 (LE target) ca 05 04 fe 00 00 c0 3f 10 5e
DATA, float 1.5 (BE target) ca 05 04 fe 3f c0 00 00 08 51
IO, channel 3 gpio_out led  ca 07 06 fe 03 00 01 6c 65 64 d3 a2
TEXT "Return value: 1\n"    ca 06 10 e9 52 65 74 75 72 6e 20 76 61 6c 75 65 3a 20 31 0a 4c 41
```

`copapy._serial.message(type, payload)` encodes a message, which is useful
for tests and other runner implementations.
