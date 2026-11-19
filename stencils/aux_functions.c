#include <stdint.h>
#include "stencil_helper.h"

volatile extern int dummy_int;
volatile extern float dummy_float;

NOINLINE float auxsub_get_42(int n) {
    return n * 5.0f + 21.0f;
}

NOINLINE float aux_get_42(float n) {
    return auxsub_get_42(n * 3.0f + 42.0f);
}

// tanh based on expf, with the Taylor series for small arguments where
// 1 - exp(-2x) loses precision
NOINLINE float aux_tanh(float x) {
    float a = fabsf(x);
    if (a < 0.25f) {
        float x2 = x * x;
        return x * (1.0f + x2 * (-1.0f / 3.0f + x2 * (2.0f / 15.0f + x2 * (-17.0f / 315.0f + x2 * (62.0f / 2835.0f)))));
    }
    float e = expf(-2.0f * a);
    float t = (1.0f - e) / (1.0f + e);
    return x < 0.0f ? -t : t;
}

// Integer power by squaring, exact up to the wrap around on overflow (like int
// multiplication). A negative exponent gives 1 / x^-e truncated towards zero:
// 0 for |x| > 1 and for x = 0
NOINLINE int aux_ipow(int x, int e) {
    if (e < 0) return x == 1 ? 1 : x == -1 ? ((e & 1) ? -1 : 1) : 0;
    unsigned int r = 1, b = (unsigned int)x;
    while (e) {
        if (e & 1) r *= b;
        b *= b;
        e >>= 1;
    }
    return (int)r;
}

// Array kernels without variants for the element types, the kernels generated
// for each type are in generate_stencils.py

// Strided copy of 32 bit elements for transposing, slicing and broadcasting.
// Parameters as int array: [offset, n0, n1, n2, n3, s0, s1, s2, s3] with the
// sizes n and the source strides s (in elements) of up to 4 dimensions.
// Contiguous rows (s3 = 1, e.g. slices of rows) are copied by a separate loop
// nest with a stride 1 inner loop that the compiler can vectorize.
#define STRIDED_COPY(type, inner_index) \
    for (int i0 = 0; i0 < n0; i0++) \
        for (int i1 = 0; i1 < n1; i1++) \
            for (int i2 = 0; i2 < n2; i2++) { \
                const type *r = s + i0 * s0 + i1 * s1 + i2 * s2; \
                for (int i3 = 0; i3 < n3; i3++) o[i3] = r[inner_index]; \
                o += n3; \
            }

KERNEL void aux_arr_copy32(const int *restrict a, int *restrict o, const int *restrict p) {
    const int *s = a + p[0];
    int n0 = p[1], n1 = p[2], n2 = p[3], n3 = p[4];
    int s0 = p[5], s1 = p[6], s2 = p[7], s3 = p[8];
    if (s3 == 1) {
        STRIDED_COPY(int, i3)
    } else {
        STRIDED_COPY(int, i3 * s3)
    }
}

// Strided copy of byte elements, parameters like for aux_arr_copy32
KERNEL void aux_arr_copy8(const byte *restrict a, byte *restrict o, const int *restrict p) {
    const byte *s = a + p[0];
    int n0 = p[1], n1 = p[2], n2 = p[3], n3 = p[4];
    int s0 = p[5], s1 = p[6], s2 = p[7], s3 = p[8];
    if (s3 == 1) {
        STRIDED_COPY(byte, i3)
    } else {
        STRIDED_COPY(byte, i3 * s3)
    }
}

#undef STRIDED_COPY

// Copy of the ring buffer a of n 32 bit elements starting at the element
// s (0 <= s < n): o[i] = a[(s + i) % n], as two contiguous copies
KERNEL void aux_arr_roll32(const int *restrict a, int *restrict o, int s, int n) {
    for (int i = 0; i < n - s; i++) o[i] = a[s + i];
    o += n - s;
    for (int i = 0; i < s; i++) o[i] = a[i];
}

// 2D convolution (cross-correlation) of a float input [n, ci, h, w] with the
// weights [co, ci, kh, kw] and a bias [co]. Parameters as int array:
// [n, ci, h, w, co, kh, kw, oh, ow, sh, sw, ph, pw, dh, dw] with the output
// size o, the strides s, the zero padding p and the dilation d. Each weight is
// accumulated over the output rows, which keeps the inner loop contiguous for
// vectorization without reassociation of floats. The loop bounds exclude the
// padding, the input is not copied.
KERNEL void aux_arr_conv2d(const float *restrict x, const float *restrict wt, const float *restrict bias, float *restrict o, const int *restrict p) {
    int nb = p[0], ci = p[1], h = p[2], w = p[3], co = p[4], kh = p[5], kw = p[6], oh = p[7], ow = p[8];
    int sh = p[9], sw = p[10], ph = p[11], pw = p[12], dh = p[13], dw = p[14];
    for (int b = 0; b < nb; b++)
        for (int c = 0; c < co; c++) {
            float *restrict oc = o + (b * co + c) * oh * ow;
            float bv = bias[c];
            for (int i = 0; i < oh * ow; i++) oc[i] = bv;
            for (int q = 0; q < ci; q++) {
                const float *xq = x + (b * ci + q) * h * w;
                const float *wq = wt + (c * ci + q) * kh * kw;
                for (int ky = 0; ky < kh; ky++) {
                    // Input row of output row oy: oy * sh - ty, must be in [0, h)
                    int ty = ph - ky * dh;
                    int y0 = ty > 0 ? (ty + sh - 1) / sh : 0;
                    int y1 = h + ty > 0 ? (h + ty - 1) / sh + 1 : 0;
                    if (y1 > oh) y1 = oh;
                    for (int kx = 0; kx < kw; kx++) {
                        int tx = pw - kx * dw;
                        int x0 = tx > 0 ? (tx + sw - 1) / sw : 0;
                        int x1 = w + tx > 0 ? (w + tx - 1) / sw + 1 : 0;
                        if (x1 > ow) x1 = ow;
                        float wv = wq[ky * kw + kx];
                        for (int oy = y0; oy < y1; oy++) {
                            const float *xr = xq + (oy * sh - ty) * w;
                            float *restrict r = oc + oy * ow;
                            for (int ox = x0; ox < x1; ox++) r[ox] += wv * xr[ox * sw - tx];
                        }
                    }
                }
            }
        }
}

// Solution x of the linear system a x = b for a row-major n x n matrix a and
// n x r right-hand sides b by Gaussian elimination with partial pivoting. The
// dimensions are passed as int array [n, r]. The elimination runs on a copy of a
// in the work array w (n x n) and on the copy of b in the result. The pivot row is
// selected by a mask and always swapped (with itself if it is already in place):
// no data dependent branches. The reciprocal of each pivot is stored in the
// diagonal of w for the back substitution (one division per column).
KERNEL void aux_arr_solve_float(const float *restrict a, const float *restrict b, float *restrict x, float *restrict w, const int *restrict p) {
    int n = p[0], r = p[1];
    for (int i = 0; i < n * n; i++) w[i] = a[i];
    for (int i = 0; i < n * r; i++) x[i] = b[i];
    for (int k = 0; k < n; k++) {
        int pv = k;
        for (int i = k + 1; i < n; i++) {
            int m = -(fabsf(w[i * n + k]) > fabsf(w[pv * n + k])); VALUE_BARRIER(m);
            pv = (i & m) | (pv & ~m);
        }
        float *wk = w + k * n, *wp = w + pv * n;
        float *xk = x + k * r, *xp = x + pv * r;
        for (int j = k; j < n; j++) { float t = wk[j]; wk[j] = wp[j]; wp[j] = t; }
        for (int j = 0; j < r; j++) { float t = xk[j]; xk[j] = xp[j]; xp[j] = t; }
        float inv = 1.0f / wk[k];
        wk[k] = inv;
        for (int i = k + 1; i < n; i++) {
            float *restrict wi = w + i * n;
            float *restrict xi = x + i * r;
            float f = wi[k] * inv;
            for (int j = k + 1; j < n; j++) wi[j] -= f * wk[j];
            for (int j = 0; j < r; j++) xi[j] -= f * xk[j];
        }
    }
    for (int i = n - 1; i >= 0; i--) {
        const float *wi = w + i * n;
        float *restrict xi = x + i * r;
        for (int q = i + 1; q < n; q++) {
            const float *xq = x + q * r;
            float f = wi[q];
            for (int j = 0; j < r; j++) xi[j] -= f * xq[j];
        }
        float inv = wi[i];
        for (int j = 0; j < r; j++) xi[j] *= inv;
    }
}

// Determinant of a row-major n x n matrix a by Gaussian elimination with partial
// pivoting on a copy of a in the work array w (n x n): the product of the pivots
// with the sign flipped for each swap of two different rows. The pivot row is
// selected and the sign is flipped without data dependent branches. A pivot of
// zero (singular matrix) is replaced by 1 as divisor: the elements below it are
// zero as well, the result is 0 instead of nan.
KERNEL float aux_arr_det_float(const float *restrict a, float *restrict w, int n) {
    union { float f; unsigned int u; } det = {1.0f};
    for (int i = 0; i < n * n; i++) w[i] = a[i];
    for (int k = 0; k < n; k++) {
        int pv = k;
        for (int i = k + 1; i < n; i++) {
            int m = -(fabsf(w[i * n + k]) > fabsf(w[pv * n + k])); VALUE_BARRIER(m);
            pv = (i & m) | (pv & ~m);
        }
        float *wk = w + k * n, *wp = w + pv * n;
        for (int j = k; j < n; j++) { float t = wk[j]; wk[j] = wp[j]; wp[j] = t; }
        det.u ^= (unsigned int)(pv != k) << 31;
        float pivot = wk[k];
        det.f *= pivot;
        float inv = 1.0f / (pivot + (float)(pivot == 0.0f));
        for (int i = k + 1; i < n; i++) {
            float *restrict wi = w + i * n;
            float f = wi[k] * inv;
            for (int j = k + 1; j < n; j++) wi[j] -= f * wk[j];
        }
    }
    return det.f;
}

// x if all bits of the int mask m are set, y if no bit is set (exact, an inf
// or nan in the operand that is not selected has no effect)
static inline float select_float(int m, float x, float y) {
    union { float f; int i; } a = {x}, b = {y}, r;
    r.i = (a.i & m) | (b.i & ~m);
    return r.f;
}

// Singular value decomposition a = u diag(s) vh of a row-major m x n matrix a by
// the one-sided Jacobi method (Hestenes) with a fixed number of sweeps, without
// data dependent branches: the execution time only depends on the size.
//
// The columns of b = a (m >= n) or b = a^T (m < n), r x k with r = max(m, n) and
// k = min(m, n), are rotated pairwise until they are orthogonal. The rotations
// accumulate to v: b = u_b diag(s) v^T with s the column norms. For a wide matrix
// a = v diag(s) u_b^T, u and vh are swapped. Each rotation puts the column with
// the larger norm first, a final sort orders the singular values descending.
//
// The matrix is scaled by a power of 2 (exact) to a largest magnitude of 1 to 2:
// the squared column norms neither overflow nor underflow for any finite input.
// Columns of u_b for singular values <= r * eps * s[0] (rank deficient matrices)
// are not defined by b: they are completed by the unit vector with the largest
// part orthogonal to the previous columns. All columns are orthogonalized (two
// Gram-Schmidt passes), u and vh are orthonormal also for singular matrices.
//
// Arguments:
//     o: Result u (m x k), s (k) and vh (k x n), row-major
//     w: Work memory of k * r + k * k floats
//     p: Parameters m, n and the number of sweeps
KERNEL void aux_arr_svd_float(const float *restrict a, float *restrict o, float *restrict w, const int *restrict p) {
    int m = p[0], n = p[1], sweeps = p[2];
    int tall = m >= n;
    int r = tall ? m : n, k = tall ? n : m;
    int sr = tall ? n : 1, sc = tall ? 1 : n;  // b[i][j] = a[i * sr + j * sc]
    float *b = w;          // k columns of b with r elements
    float *v = w + k * r;  // k columns of v with k elements
    float *s = o + m * k;

    // Exact scaling by 2^(127 - e) for a largest magnitude of 2^(e - 127) * 1.x
    union { float f; unsigned int u; } amax = {0.0f}, scale, unscale;
    for (int i = 0; i < m * n; i++) amax.f = select_float(-(fabsf(a[i]) > amax.f), fabsf(a[i]), amax.f);
    int e = (int)(amax.u >> 23);  // Sign bit is 0
    e = e < 2 ? 2 : e > 253 ? 253 : e;
    scale.u = (unsigned int)(254 - e) << 23;
    unscale.u = (unsigned int)e << 23;

    for (int j = 0; j < k; j++) {
        for (int i = 0; i < r; i++) b[j * r + i] = a[i * sr + j * sc] * scale.f;
        for (int i = 0; i < k; i++) v[j * k + i] = (float)(i == j);
    }

    for (int sweep = 0; sweep < sweeps; sweep++) {
        for (int pi = 0; pi < k - 1; pi++) {
            for (int qi = pi + 1; qi < k; qi++) {
                float *restrict bp = b + pi * r, *restrict bq = b + qi * r;
                float *restrict vp = v + pi * k, *restrict vq = v + qi * k;
                float alpha = 0.0f, beta = 0.0f, gamma = 0.0f;
                for (int i = 0; i < r; i++) {
                    alpha += bp[i] * bp[i];
                    beta += bq[i] * bq[i];
                    gamma += bp[i] * bq[i];
                }
                // tan of the rotation angle t = sign(z) / (|z| + sqrt(1 + z^2)) with
                // z = (beta - alpha) / (2 gamma), without a division by gamma: t = 0
                // for gamma = 0 (orthogonal columns), t = +-1 for alpha = beta
                float d = beta - alpha;
                float den = fabsf(d) + sqrtf(d * d + 4.0f * gamma * gamma);
                den += (float)(den == 0.0f);
                float t = 2.0f * gamma * copysignf(1.0f, d) / den;
                float c = 1.0f / sqrtf(1.0f + t * t);
                float sn = c * t;
                // Norms after the rotation: alpha - t gamma and beta + t gamma,
                // swap the new columns if the second one is larger
                int sw = -((alpha - t * gamma) < (beta + t * gamma)); VALUE_BARRIER(sw);
                float c1 = select_float(sw, sn, c), s1 = select_float(sw, c, -sn);
                float c2 = select_float(sw, c, sn), s2 = select_float(sw, -sn, c);
                for (int i = 0; i < r; i++) {
                    float x = bp[i], y = bq[i];
                    bp[i] = c1 * x + s1 * y;
                    bq[i] = c2 * x + s2 * y;
                }
                for (int i = 0; i < k; i++) {
                    float x = vp[i], y = vq[i];
                    vp[i] = c1 * x + s1 * y;
                    vq[i] = c2 * x + s2 * y;
                }
            }
        }
    }

    // Singular values sorted descending: selection sort, swaps of the columns
    for (int j = 0; j < k; j++) {
        float acc = 0.0f;
        for (int i = 0; i < r; i++) acc += b[j * r + i] * b[j * r + i];
        s[j] = sqrtf(acc);
    }
    for (int j = 0; j < k; j++) {
        int pv = j;
        for (int i = j + 1; i < k; i++) {
            int msk = -(s[i] > s[pv]); VALUE_BARRIER(msk);
            pv = (i & msk) | (pv & ~msk);
        }
        float t = s[j]; s[j] = s[pv]; s[pv] = t;
        float *bj = b + j * r, *bv = b + pv * r, *vj = v + j * k, *vv = v + pv * k;
        for (int i = 0; i < r; i++) { float x = bj[i]; bj[i] = bv[i]; bv[i] = x; }
        for (int i = 0; i < k; i++) { float x = vj[i]; vj[i] = vv[i]; vv[i] = x; }
    }

    // Columns of u_b: normalized columns of b or completed for singular values
    // close to zero, orthogonalized against the previous columns
    float tol = s[0] * (float)r * 1.1920929e-7f;
    for (int j = 0; j < k; j++) {
        float *restrict bj = b + j * r;
        int keep = -(s[j] > tol); VALUE_BARRIER(keep);
        int best_i = 0;
        float best = -1.0f;
        for (int i = 0; i < r; i++) {
            float res = 1.0f;  // Squared norm of the unit vector i orthogonal to the previous columns
            for (int l = 0; l < j; l++) res -= b[l * r + i] * b[l * r + i];
            int msk = -(res > best); VALUE_BARRIER(msk);
            best_i = (i & msk) | (best_i & ~msk);
            best = select_float(msk, res, best);
        }
        float inv = 1.0f / s[j];  // inf for s[j] = 0, not selected then
        for (int i = 0; i < r; i++) bj[i] = select_float(keep, bj[i] * inv, (float)(i == best_i));
        for (int pass = 0; pass < 2; pass++) {
            for (int l = 0; l < j; l++) {
                const float *bl = b + l * r;
                float dl = 0.0f;
                for (int i = 0; i < r; i++) dl += bl[i] * bj[i];
                for (int i = 0; i < r; i++) bj[i] -= dl * bl[i];
            }
        }
        float acc = 0.0f;
        for (int i = 0; i < r; i++) acc += bj[i] * bj[i];
        float nrm = 1.0f / sqrtf(acc);
        for (int i = 0; i < r; i++) bj[i] *= nrm;
    }

    for (int j = 0; j < k; j++) s[j] *= unscale.f;

    // u: m x k, vh: k x n
    float *vh = s + k;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++) o[i * k + j] = tall ? b[j * r + i] : v[j * k + i];
    for (int j = 0; j < k; j++)
        for (int i = 0; i < n; i++) vh[j * n + i] = tall ? v[j * k + i] : b[j * r + i];
}

// Value of the table fp (n >= 2 elements) at the fractional index t by linear
// interpolation between the elements floor(t) and floor(t) + 1. The index is
// clamped to the table: t <= 0 results in fp[0], t >= n - 1 in fp[n - 1]. No
// element outside of the table is accessed for any t, nan results in nan.
static inline float lerp_index(const float *fp, float t, int n) {
    float last = (float)(n - 1);
    float tn = t < 0.0f ? 0.0f : t;  // keeps nan
    tn = tn > last ? last : tn;
    float ts = t > 0.0f ? t : 0.0f;  // nan to 0 for the element index
    ts = ts < last ? ts : last;
    int i = (int)ts;
    i = i < n - 2 ? i : n - 2;
    float f = tn - (float)i;
    return fp[i] * (1.0f - f) + fp[i + 1] * f;
}

KERNEL float aux_arr_lerp_float(const float *restrict fp, float t, int n) {
    return lerp_index(fp, t, n);
}

KERNEL void aux_arr_lerp_floatarr(const float *restrict fp, const float *restrict t, float *restrict o, int n, int m) {
    for (int j = 0; j < m; j++) o[j] = lerp_index(fp, t[j], n);
}

// Fractional index of x in the increasing grid xp (n >= 2 elements): the index i of
// the interval xp[i] <= x < xp[i + 1] plus the position of x in the interval. The
// result is clamped to [0, n - 1] for x outside of the grid, nan results in nan.
// The interval is found by a binary search without data dependent branches, the
// number of steps only depends on n.
static inline float search_index(const float *xp, float x, int n) {
    int lo = 0;
    for (int len = n - 1; len > 1;) {
        int half = len >> 1;
        int m = -(xp[lo + half] <= x); VALUE_BARRIER(m);
        lo += half & m;
        len -= half;
    }
    // Position in the interval clamped to [0, 1] by masks, nan is kept
    union { float f; int i; } u = {(x - xp[lo]) / (xp[lo + 1] - xp[lo])};
    int below = -(u.f < 0.0f); VALUE_BARRIER(below);
    u.i &= ~below;
    float f = u.f > 1.0f ? 1.0f : u.f;
    return (float)lo + f;
}

KERNEL float aux_arr_bsearch_float(const float *restrict xp, float x, int n) {
    return search_index(xp, x, n);
}

KERNEL void aux_arr_bsearch_floatarr(const float *restrict xp, const float *restrict x, float *restrict o, int n, int m) {
    for (int j = 0; j < m; j++) o[j] = search_index(xp, x[j], n);
}

#if defined(__tricore__)
/* Math functions of newlib (used for TriCore) set errno, it is
   initialized to be placed in .data, since .bss is not allocated */
static int errno_placeholder = 1;

NOINLINE int *__errno(void) {
    return &errno_placeholder;
}
#endif
