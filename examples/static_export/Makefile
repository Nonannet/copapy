# Static export example: firmware for a Cortex-M4 with a copapy program
# that is traced and compiled by make and linked as static part.

# Addresses of the copapy program: used to generate the program and by the
# linker script. Both must be aligned to 16 bytes.
COPAPY_CODE_ADDR ?= 0x00100000
COPAPY_DATA_ADDR ?= 0x20100000
COPAPY_ARCH ?= armv7mthumb

PYTHON ?= python3
CROSS ?= arm-none-eabi-
QEMU ?= qemu-system-arm

BUILD := build
GEN := $(BUILD)/gen

CPU_FLAGS := -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard
CFLAGS := $(CPU_FLAGS) -O2 -g -Wall -Wextra -ffunction-sections -fdata-sections -I$(GEN)
LDFLAGS := $(CPU_FLAGS) -nostartfiles -T link.ld -Wl,--gc-sections -Wl,-Map=$(BUILD)/firmware.map \
	-Wl,--defsym=COPAPY_CODE_ADDR=$(COPAPY_CODE_ADDR) \
	-Wl,--defsym=COPAPY_DATA_ADDR=$(COPAPY_DATA_ADDR)

OBJS := $(BUILD)/startup.o $(BUILD)/main.o $(BUILD)/control.o

all: $(BUILD)/firmware.elf

# Trace and compile the copapy program, writes control.c and control.h
$(GEN)/control.c: control.py Makefile
	$(PYTHON) control.py --out $(GEN) --arch $(COPAPY_ARCH) \
		--code-addr $(COPAPY_CODE_ADDR) --data-addr $(COPAPY_DATA_ADDR)

$(GEN)/control.h: $(GEN)/control.c ;

$(BUILD)/control.o: $(GEN)/control.c $(GEN)/control.h
	$(CROSS)gcc $(CFLAGS) -c $< -o $@

$(BUILD)/main.o: $(GEN)/control.h

$(BUILD)/%.o: %.c semihost.h | $(BUILD)
	$(CROSS)gcc $(CFLAGS) -c $< -o $@

$(BUILD)/firmware.elf: $(OBJS) link.ld
	$(CROSS)gcc $(LDFLAGS) $(OBJS) -o $@
	$(CROSS)size $@

$(BUILD):
	mkdir -p $(BUILD)

# Run the firmware in QEMU, the output is written by semihosting
run: $(BUILD)/firmware.elf
	$(QEMU) -M mps2-an386 -nographic -semihosting-config enable=on,target=native -kernel $<

clean:
	rm -rf $(BUILD)

.PHONY: all run clean
