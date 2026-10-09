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

// Array kernels without variants for the element types, the kernels generated
// for each type are in generate_stencils.py

// Strided copy of 32 bit elements for transposing, slicing and broadcasting.
// Parameters as int array: [offset, n0, n1, n2, n3, s0, s1, s2, s3] with the
// sizes n and the source strides s (in elements) of up to 4 dimensions.
KERNEL void aux_arr_copy32(const int *restrict a, int *restrict o, const int *restrict p) {
    const int *s = a + p[0];
    int n0 = p[1], n1 = p[2], n2 = p[3], n3 = p[4];
    int s0 = p[5], s1 = p[6], s2 = p[7], s3 = p[8];
    for (int i0 = 0; i0 < n0; i0++)
        for (int i1 = 0; i1 < n1; i1++)
            for (int i2 = 0; i2 < n2; i2++) {
                const int *r = s + i0 * s0 + i1 * s1 + i2 * s2;
                for (int i3 = 0; i3 < n3; i3++) *o++ = r[i3 * s3];
            }
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
