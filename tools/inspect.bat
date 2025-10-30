python tools/make_example.py
python tools/extract_code.py "bin/test-aarch64.copapy" "bin/test-aarch64.copapy.bin"
wsl aarch64-linux-gnu-objdump -D -b binary -m aarch64 --adjust-vma=0x1000 bin/test-aarch64.copapy.bin
