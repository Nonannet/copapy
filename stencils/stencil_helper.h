#include <math.h>

// Remove function alignment for stencils
#if defined(__GNUC__)
#define NOINLINE __attribute__((noinline))
#define STENCIL __attribute__((aligned(1)))
#else
#define NOINLINE
#define STENCIL
#endif