python tools\tricore\make_trc_example.py
build\runner\coparun build/runner/test-tricore.copapy build/runner/test-tricore.copapy.bin
python .\tools\tricore\run_qemu.py --code build/runner/test-tricore.copapy.bin