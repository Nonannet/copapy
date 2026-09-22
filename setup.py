from pathlib import Path
import importlib.util
import os

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


ROOT = Path(__file__).resolve().parent


class BuildExt(build_ext):

    def build_extension(self, ext):
        if os.name == "nt":
            # On Windows reassemble the ABI shim object from annotated hex dump
            path = ROOT / "tools" / "hexdump_annotated.py"
            spec = importlib.util.spec_from_file_location("hexdump_annotated", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            shim_hex = ROOT / "src" / "coparun" / "x86_64_abi_shim.hex"

            obj_dir = Path(self.build_temp)
            obj_dir.mkdir(parents=True, exist_ok=True)
            obj = obj_dir / "x86_64_abi_shim.obj"

            print(f"shim: reassembling {shim_hex.name} -> {obj.name}")

            if mod.reassemble(str(shim_hex), str(obj)) != 0:
                raise RuntimeError(
                    f"Failed to reassemble {obj.name} from {shim_hex.name}"
                )

            ext.extra_objects = [*(ext.extra_objects or []), str(obj)]

        super().build_extension(ext)


ext = Extension(
    "coparun_module",
    sources=[
        "src/coparun/coparun_module.c",
        "src/coparun/runmem.c",
        "src/coparun/mem_man.c",
    ],
)


setup(
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExt},
)
