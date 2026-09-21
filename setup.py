from pathlib import Path
import os
import shutil
import subprocess

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


ROOT = Path(__file__).resolve().parent


def find_ml64():
    # 1. Already configured MSVC environment.
    ml64 = shutil.which("ml64.exe")
    if ml64:
        return Path(ml64)

    # 2. Find Visual Studio using vswhere.
    vswhere = (
        Path(os.environ.get("ProgramFiles(x86)", ""))
        / "Microsoft Visual Studio"
        / "Installer"
        / "vswhere.exe"
    )

    if not vswhere.is_file():
        raise RuntimeError(
            "Could not find vswhere.exe. "
            "Install Visual Studio or the Visual Studio Build Tools."
        )

    result = subprocess.run(
        [
            str(vswhere),
            "-latest",
            "-products", "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    vs_path = Path(result.stdout.strip())

    if not vs_path:
        raise RuntimeError(
            "Could not find a Visual Studio installation containing "
            "the MSVC x64/x86 tools."
        )

    msvc_root = vs_path / "VC" / "Tools" / "MSVC"

    # Use the toolset Visual Studio considers the default.
    version_file = (
        vs_path
        / "VC"
        / "Auxiliary"
        / "Build"
        / "Microsoft.VCToolsVersion.default.txt"
    )

    if version_file.is_file():
        version = version_file.read_text().strip()
        msvc = msvc_root / version
    else:
        versions = sorted(
            p for p in msvc_root.iterdir()
            if p.is_dir()
        )

        if not versions:
            raise RuntimeError(
                f"No MSVC toolset found in {msvc_root}"
            )

        msvc = versions[-1]

    ml64 = msvc / "bin" / "Hostx64" / "x64" / "ml64.exe"

    if not ml64.is_file():
        raise RuntimeError(
            f"Could not find ml64.exe at {ml64}"
        )

    return ml64


class BuildExt(build_ext):

    def build_extension(self, ext):
        if os.name == "nt":
            self.build_asm(ext)

        super().build_extension(ext)

    def build_asm(self, ext):
        ml64 = find_ml64()

        asm = ROOT / "src" / "coparun" / "x86_64_abi_shim.asm"

        obj_dir = Path(self.build_temp)
        obj_dir.mkdir(parents=True, exist_ok=True)

        obj = obj_dir / "x86_64_abi_shim.obj"

        print(f"MASM: {asm.name}")

        subprocess.run(
            [
                str(ml64),
                "/nologo",
                "/c",
                f"/Fo{obj}",
                str(asm),
            ],
            check=True,
        )

        ext.extra_objects = [
            *(ext.extra_objects or []),
            str(obj),
        ]


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
