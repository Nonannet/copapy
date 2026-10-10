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

// Array stencils: operands are heap objects referenced by the ref_* symbols,
// patched by the compiler to the addresses of the node arguments (ref_arg<n>)
// and the result (ref_out). All heap objects are at least 4 byte aligned.
#define REF_OBJ(sym) extern char sym[] __attribute__((aligned(4)))

// Element type of byte arrays (e.g. image data), bytes are only stored:
// for operations the arrays are converted to int or float
typedef unsigned char byte;
REF_OBJ(ref_arg0);
REF_OBJ(ref_arg1);
REF_OBJ(ref_arg2);
REF_OBJ(ref_arg3);
REF_OBJ(ref_out);

// Address of a ref_* symbol. On x86_64 -fno-pic takes addresses of extern
// objects as 32 bit absolute values (R_X86_64_32), force RIP relative addressing
#if defined(__x86_64__)
#define REF(sym) ({ void *p_; __asm__("lea " #sym "(%%rip), %0" : "=r"(p_)); p_; })
#else
#define REF(sym) ((void *)(sym))
#endif

// Hide a value from the optimizer (no instruction): prevents gcc from turning
// branch-free code (e.g. a swap by a mask from a comparison) back into branches
#if defined(__GNUC__)
#define VALUE_BARRIER(x) __asm__("" : "+r"(x))
#else
#define VALUE_BARRIER(x)
#endif

// Array kernels are shared auxiliary functions called as normal function by the
// array stencils. Loops must not be replaced by memcpy/memset calls (no libc).
#if defined(__GNUC__) && !defined(__clang__)
#define NO_LIBCALLS optimize("no-tree-loop-distribute-patterns")
#else
#define NO_LIBCALLS
#endif
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#define KERNEL __attribute__((noinline, force_align_arg_pointer, NO_LIBCALLS))
#elif defined(__GNUC__)
#define KERNEL __attribute__((noinline, NO_LIBCALLS))
#else
#define KERNEL NOINLINE
#endif