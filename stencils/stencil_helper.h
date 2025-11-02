#if defined(__GNUC__)
#define NOINLINE __attribute__((noinline))
#define STENCIL_START_EX(funcname) \
    __asm__ __volatile__( \
        ".global stencil_start_" #funcname "\n" \
        "stencil_start_" #funcname ":\n" \
    )
#define STENCIL_START(funcname)
#else
#define NOINLINE
#define STENCIL_START(funcname)
#endif