from setuptools import setup, Extension

ext = Extension(
    "coparun_module",
    sources=[
        "src/coparun/coparun_module.c",
        "src/coparun/runmem.c"
    ]
)

setup(
    ext_modules=[ext],
)
