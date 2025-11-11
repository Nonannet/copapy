//#include <stdint.h>

typedef float float_t;
typedef double double_t;
typedef unsigned int uint32_t;
typedef double double_t;
typedef unsigned short int uint16_t;
typedef unsigned long long uint64_t;

#define M_E             2.7182818284590452354   /* e */
#define M_LOG2E         1.4426950408889634074   /* log_2 e */
#define M_LOG10E        0.43429448190325182765  /* log_10 e */
#define M_LN2           0.69314718055994530942  /* log_e 2 */
#define M_LN10          2.30258509299404568402  /* log_e 10 */
#define M_PI            3.14159265358979323846  /* pi */
#define M_PI_2          1.57079632679489661923  /* pi/2 */
#define M_PI_4          0.78539816339744830962  /* pi/4 */
#define M_1_PI          0.31830988618379067154  /* 1/pi */
#define M_2_PI          0.63661977236758134308  /* 2/pi */
#define M_2_SQRTPI      1.12837916709551257390  /* 2/sqrt(pi) */
#define M_SQRT2         1.41421356237309504880  /* sqrt(2) */
#define M_SQRT1_2       0.70710678118654752440  /* 1/sqrt(2) */

float  __sindf(double);
float  __cosdf(double);
float  __tandf(double,int);

const uint16_t __rsqrt_tab[128] = {
    0xb451,0xb2f0,0xb196,0xb044,0xaef9,0xadb6,0xac79,0xab43,
    0xaa14,0xa8eb,0xa7c8,0xa6aa,0xa592,0xa480,0xa373,0xa26b,
    0xa168,0xa06a,0x9f70,0x9e7b,0x9d8a,0x9c9d,0x9bb5,0x9ad1,
    0x99f0,0x9913,0x983a,0x9765,0x9693,0x95c4,0x94f8,0x9430,
    0x936b,0x92a9,0x91ea,0x912e,0x9075,0x8fbe,0x8f0a,0x8e59,
    0x8daa,0x8cfe,0x8c54,0x8bac,0x8b07,0x8a64,0x89c4,0x8925,
    0x8889,0x87ee,0x8756,0x86c0,0x862b,0x8599,0x8508,0x8479,
    0x83ec,0x8361,0x82d8,0x8250,0x81c9,0x8145,0x80c2,0x8040,
    0xff02,0xfd0e,0xfb25,0xf947,0xf773,0xf5aa,0xf3ea,0xf234,
    0xf087,0xeee3,0xed47,0xebb3,0xea27,0xe8a3,0xe727,0xe5b2,
    0xe443,0xe2dc,0xe17a,0xe020,0xdecb,0xdd7d,0xdc34,0xdaf1,
    0xd9b3,0xd87b,0xd748,0xd61a,0xd4f1,0xd3cd,0xd2ad,0xd192,
    0xd07b,0xcf69,0xce5b,0xcd51,0xcc4a,0xcb48,0xca4a,0xc94f,
    0xc858,0xc764,0xc674,0xc587,0xc49d,0xc3b7,0xc2d4,0xc1f4,
    0xc116,0xc03c,0xbf65,0xbe90,0xbdbe,0xbcef,0xbc23,0xbb59,
    0xba91,0xb9cc,0xb90a,0xb84a,0xb78c,0xb6d0,0xb617,0xb560,
    };

/* fp_force_eval ensures that the input value is computed when that's
   otherwise unused.  To prevent the constant folding of the input
   expression, an additional fp_barrier may be needed or a compilation
   mode that does so (e.g. -frounding-math in gcc). Then it can be
   used to evaluate an expression for its fenv side-effects only.   */

#ifndef fp_force_evalf
#define fp_force_evalf fp_force_evalf
static inline void fp_force_evalf(float x)
{
    volatile float y;
    y = x;
}
#endif

#ifndef fp_force_eval
#define fp_force_eval fp_force_eval
static inline void fp_force_eval(double x)
{
    volatile double y;
    y = x;
}
#endif

#ifndef fp_force_evall
#define fp_force_evall fp_force_evall
static inline void fp_force_evall(long double x)
{
    volatile long double y;
    y = x;
}
#endif

#define FORCE_EVAL(x) do {                        \
    if (sizeof(x) == sizeof(float)) {         \
        fp_force_evalf(x);                \
    } else if (sizeof(x) == sizeof(double)) { \
        fp_force_eval(x);                 \
    } else {                                  \
        fp_force_evall(x);                \
    }                                         \
} while(0)

#define asfloat(i) ((union{uint32_t _i; float _f;}){i})._f
#define asuint(f) ((union{float _f; uint32_t _i;}){f})._i

/* Evaluate an expression as the specified type. With standard excess
   precision handling a type cast or assignment is enough (with
   -ffloat-store an assignment is required, in old compilers argument
   passing and return statement may not drop excess precision).  */

static inline float eval_as_float(float x)
{
	float y = x;
	return y;
}

/* Helps static branch prediction so hot path can be better optimized.  */
#ifdef __GNUC__
#define predict_true(x) __builtin_expect(!!(x), 1)
#define predict_false(x) __builtin_expect(x, 0)
#else
#define predict_true(x) (x)
#define predict_false(x) (x)
#endif

/* error handling functions */
#define __math_invalidf(x) (x - x) / (x - x)