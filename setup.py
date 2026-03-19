from setuptools import setup, Extension

ext = Extension(
    "coparun_module",
    sources=[
        "src/coparun/coparun_module.c",
        "src/coparun/runmem.c",
        "src/coparun/mem_man.c"
    ],
    #define_macros=[("ENABLE_BASIC_LOGGING", None)]
)

setup(
    ext_modules=[ext],
)
