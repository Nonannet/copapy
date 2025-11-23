#include <math.h>

// Remove function alignment for stencils
#if defined(__GNUC__)
#define NOINLINE __attribute__((noinline))
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(__thumb__) || defined(_M_ARM)
#define STENCIL __attribute__((aligned(4)))
#else
#define STENCIL __attribute__((aligned(1)))
#endif
#else
#define NOINLINE
#define STENCIL
#endif