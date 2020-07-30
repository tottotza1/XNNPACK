// Auto-generated file. Do not edit!
//   Template: src/qu8-gemm/MRxNRc4-minmax-scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/gemm.h>

#include <xnnpack/scalar-utils.h>

// This kernel is a scalar model for a kernel using ARMv8.2 dot-product
// instructions.
//
// XNN_DISABLE_TSAN is used because this kernel reads up to 3 bytes past the
// bounds of the `a` matrix region, which may be a race condition with
// another thread. We deem this acceptable because the values that are
// read out of bounds do not affect the result, and the the compiler can't know
// about this undefined behavior.
void xnn_qu8_gemm_minmax_ukernel_12x4c4__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    const uint8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qu8_gemm_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN {
  assert(mr != 0);
  assert(mr <= 12);
  assert(nc != 0);
  assert(kc != 0);

  const uint8_t* a0 = a;
  uint8_t* c0 = c;
  const uint8_t* a1 = (const uint8_t*) ((uintptr_t) a0 + a_stride);
  uint8_t* c1 = (uint8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const uint8_t* a2 = (const uint8_t*) ((uintptr_t) a1 + a_stride);
  uint8_t* c2 = (uint8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const uint8_t* a3 = (const uint8_t*) ((uintptr_t) a2 + a_stride);
  uint8_t* c3 = (uint8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const uint8_t* a4 = (const uint8_t*) ((uintptr_t) a3 + a_stride);
  uint8_t* c4 = (uint8_t*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const uint8_t* a5 = (const uint8_t*) ((uintptr_t) a4 + a_stride);
  uint8_t* c5 = (uint8_t*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    a5 = a4;
    c5 = c4;
  }
  const uint8_t* a6 = (const uint8_t*) ((uintptr_t) a5 + a_stride);
  uint8_t* c6 = (uint8_t*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    a6 = a5;
    c6 = c5;
  }
  const uint8_t* a7 = (const uint8_t*) ((uintptr_t) a6 + a_stride);
  uint8_t* c7 = (uint8_t*) ((uintptr_t) c6 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 8) {
    a7 = a6;
    c7 = c6;
  }
  const uint8_t* a8 = (const uint8_t*) ((uintptr_t) a7 + a_stride);
  uint8_t* c8 = (uint8_t*) ((uintptr_t) c7 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 8) {
    a8 = a7;
    c8 = c7;
  }
  const uint8_t* a9 = (const uint8_t*) ((uintptr_t) a8 + a_stride);
  uint8_t* c9 = (uint8_t*) ((uintptr_t) c8 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 10) {
    a9 = a8;
    c9 = c8;
  }
  const uint8_t* a10 = (const uint8_t*) ((uintptr_t) a9 + a_stride);
  uint8_t* c10 = (uint8_t*) ((uintptr_t) c9 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 10) {
    a10 = a9;
    c10 = c9;
  }
  const uint8_t* a11 = (const uint8_t*) ((uintptr_t) a10 + a_stride);
  uint8_t* c11 = (uint8_t*) ((uintptr_t) c10 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 12) {
    a11 = a10;
    c11 = c10;
  }

  const int32_t vb_zero_point = params->scalar.kernel_zero_point;

  // Loop over groups of 4 columns.
  do {
    // `vaccMN` is the accumulator at row `M`, column `N`.
    // Initialize accumulators with bias. 4 bias values are loaded from the
    // weight matrix, at the start of the group of 4 columns.
    int32_t bias0 = ((const int32_t*)w)[0];
    int32_t vacc00 = bias0;
    int32_t vacc10 = bias0;
    int32_t vacc20 = bias0;
    int32_t vacc30 = bias0;
    int32_t vacc40 = bias0;
    int32_t vacc50 = bias0;
    int32_t vacc60 = bias0;
    int32_t vacc70 = bias0;
    int32_t vacc80 = bias0;
    int32_t vacc90 = bias0;
    int32_t vacc100 = bias0;
    int32_t vacc110 = bias0;
    int32_t bias1 = ((const int32_t*)w)[1];
    int32_t vacc01 = bias1;
    int32_t vacc11 = bias1;
    int32_t vacc21 = bias1;
    int32_t vacc31 = bias1;
    int32_t vacc41 = bias1;
    int32_t vacc51 = bias1;
    int32_t vacc61 = bias1;
    int32_t vacc71 = bias1;
    int32_t vacc81 = bias1;
    int32_t vacc91 = bias1;
    int32_t vacc101 = bias1;
    int32_t vacc111 = bias1;
    int32_t bias2 = ((const int32_t*)w)[2];
    int32_t vacc02 = bias2;
    int32_t vacc12 = bias2;
    int32_t vacc22 = bias2;
    int32_t vacc32 = bias2;
    int32_t vacc42 = bias2;
    int32_t vacc52 = bias2;
    int32_t vacc62 = bias2;
    int32_t vacc72 = bias2;
    int32_t vacc82 = bias2;
    int32_t vacc92 = bias2;
    int32_t vacc102 = bias2;
    int32_t vacc112 = bias2;
    int32_t bias3 = ((const int32_t*)w)[3];
    int32_t vacc03 = bias3;
    int32_t vacc13 = bias3;
    int32_t vacc23 = bias3;
    int32_t vacc33 = bias3;
    int32_t vacc43 = bias3;
    int32_t vacc53 = bias3;
    int32_t vacc63 = bias3;
    int32_t vacc73 = bias3;
    int32_t vacc83 = bias3;
    int32_t vacc93 = bias3;
    int32_t vacc103 = bias3;
    int32_t vacc113 = bias3;

    w = (const void*)((uintptr_t)w + 4 * sizeof(int32_t));

    // Inner accumulation loop along the 4 columns.
    // Handle 4 rows at each iteration: this is key to modelling what an
    // actual kernel using ARMv8.2 dot-product instructions would look like.
    size_t k = 0;
    while (k < kc) {
      // Load a 12x4 block of activations, and compute sums along rows.
      int16_t vasum0 = 0;
      uint8_t va00 = *a0++;
      vasum0 += (int16_t) va00;
      uint8_t va01 = *a0++;
      vasum0 += (int16_t) va01;
      uint8_t va02 = *a0++;
      vasum0 += (int16_t) va02;
      uint8_t va03 = *a0++;
      vasum0 += (int16_t) va03;
      int16_t vasum1 = 0;
      uint8_t va10 = *a1++;
      vasum1 += (int16_t) va10;
      uint8_t va11 = *a1++;
      vasum1 += (int16_t) va11;
      uint8_t va12 = *a1++;
      vasum1 += (int16_t) va12;
      uint8_t va13 = *a1++;
      vasum1 += (int16_t) va13;
      int16_t vasum2 = 0;
      uint8_t va20 = *a2++;
      vasum2 += (int16_t) va20;
      uint8_t va21 = *a2++;
      vasum2 += (int16_t) va21;
      uint8_t va22 = *a2++;
      vasum2 += (int16_t) va22;
      uint8_t va23 = *a2++;
      vasum2 += (int16_t) va23;
      int16_t vasum3 = 0;
      uint8_t va30 = *a3++;
      vasum3 += (int16_t) va30;
      uint8_t va31 = *a3++;
      vasum3 += (int16_t) va31;
      uint8_t va32 = *a3++;
      vasum3 += (int16_t) va32;
      uint8_t va33 = *a3++;
      vasum3 += (int16_t) va33;
      int16_t vasum4 = 0;
      uint8_t va40 = *a4++;
      vasum4 += (int16_t) va40;
      uint8_t va41 = *a4++;
      vasum4 += (int16_t) va41;
      uint8_t va42 = *a4++;
      vasum4 += (int16_t) va42;
      uint8_t va43 = *a4++;
      vasum4 += (int16_t) va43;
      int16_t vasum5 = 0;
      uint8_t va50 = *a5++;
      vasum5 += (int16_t) va50;
      uint8_t va51 = *a5++;
      vasum5 += (int16_t) va51;
      uint8_t va52 = *a5++;
      vasum5 += (int16_t) va52;
      uint8_t va53 = *a5++;
      vasum5 += (int16_t) va53;
      int16_t vasum6 = 0;
      uint8_t va60 = *a6++;
      vasum6 += (int16_t) va60;
      uint8_t va61 = *a6++;
      vasum6 += (int16_t) va61;
      uint8_t va62 = *a6++;
      vasum6 += (int16_t) va62;
      uint8_t va63 = *a6++;
      vasum6 += (int16_t) va63;
      int16_t vasum7 = 0;
      uint8_t va70 = *a7++;
      vasum7 += (int16_t) va70;
      uint8_t va71 = *a7++;
      vasum7 += (int16_t) va71;
      uint8_t va72 = *a7++;
      vasum7 += (int16_t) va72;
      uint8_t va73 = *a7++;
      vasum7 += (int16_t) va73;
      int16_t vasum8 = 0;
      uint8_t va80 = *a8++;
      vasum8 += (int16_t) va80;
      uint8_t va81 = *a8++;
      vasum8 += (int16_t) va81;
      uint8_t va82 = *a8++;
      vasum8 += (int16_t) va82;
      uint8_t va83 = *a8++;
      vasum8 += (int16_t) va83;
      int16_t vasum9 = 0;
      uint8_t va90 = *a9++;
      vasum9 += (int16_t) va90;
      uint8_t va91 = *a9++;
      vasum9 += (int16_t) va91;
      uint8_t va92 = *a9++;
      vasum9 += (int16_t) va92;
      uint8_t va93 = *a9++;
      vasum9 += (int16_t) va93;
      int16_t vasum10 = 0;
      uint8_t va100 = *a10++;
      vasum10 += (int16_t) va100;
      uint8_t va101 = *a10++;
      vasum10 += (int16_t) va101;
      uint8_t va102 = *a10++;
      vasum10 += (int16_t) va102;
      uint8_t va103 = *a10++;
      vasum10 += (int16_t) va103;
      int16_t vasum11 = 0;
      uint8_t va110 = *a11++;
      vasum11 += (int16_t) va110;
      uint8_t va111 = *a11++;
      vasum11 += (int16_t) va111;
      uint8_t va112 = *a11++;
      vasum11 += (int16_t) va112;
      uint8_t va113 = *a11++;
      vasum11 += (int16_t) va113;

      // Load a 4x4 block of weights.
      uint8_t vb00 = ((const uint8_t*)w)[0];
      uint8_t vb10 = ((const uint8_t*)w)[1];
      uint8_t vb20 = ((const uint8_t*)w)[2];
      uint8_t vb30 = ((const uint8_t*)w)[3];

      w = (const void*)((uintptr_t)w + 4 * sizeof(uint8_t));
      uint8_t vb01 = ((const uint8_t*)w)[0];
      uint8_t vb11 = ((const uint8_t*)w)[1];
      uint8_t vb21 = ((const uint8_t*)w)[2];
      uint8_t vb31 = ((const uint8_t*)w)[3];

      w = (const void*)((uintptr_t)w + 4 * sizeof(uint8_t));
      uint8_t vb02 = ((const uint8_t*)w)[0];
      uint8_t vb12 = ((const uint8_t*)w)[1];
      uint8_t vb22 = ((const uint8_t*)w)[2];
      uint8_t vb32 = ((const uint8_t*)w)[3];

      w = (const void*)((uintptr_t)w + 4 * sizeof(uint8_t));
      uint8_t vb03 = ((const uint8_t*)w)[0];
      uint8_t vb13 = ((const uint8_t*)w)[1];
      uint8_t vb23 = ((const uint8_t*)w)[2];
      uint8_t vb33 = ((const uint8_t*)w)[3];

      w = (const void*)((uintptr_t)w + 4 * sizeof(uint8_t));

      // Multiply-accumulate: 12x4 * 4x4 --> 12x4. The inner size 4 here means
      // we're computing 4D dot-products, which makes this a model for
      // a ARMv8.2 dot-product kernel.
      vacc00 += va00 * vb00;
      vacc00 += va01 * vb10;
      vacc00 += va02 * vb20;
      vacc00 += va03 * vb30;
      vacc00 -= ((int32_t) vasum0) * vb_zero_point;
      vacc01 += va00 * vb01;
      vacc01 += va01 * vb11;
      vacc01 += va02 * vb21;
      vacc01 += va03 * vb31;
      vacc01 -= ((int32_t) vasum0) * vb_zero_point;
      vacc02 += va00 * vb02;
      vacc02 += va01 * vb12;
      vacc02 += va02 * vb22;
      vacc02 += va03 * vb32;
      vacc02 -= ((int32_t) vasum0) * vb_zero_point;
      vacc03 += va00 * vb03;
      vacc03 += va01 * vb13;
      vacc03 += va02 * vb23;
      vacc03 += va03 * vb33;
      vacc03 -= ((int32_t) vasum0) * vb_zero_point;
      vacc10 += va10 * vb00;
      vacc10 += va11 * vb10;
      vacc10 += va12 * vb20;
      vacc10 += va13 * vb30;
      vacc10 -= ((int32_t) vasum1) * vb_zero_point;
      vacc11 += va10 * vb01;
      vacc11 += va11 * vb11;
      vacc11 += va12 * vb21;
      vacc11 += va13 * vb31;
      vacc11 -= ((int32_t) vasum1) * vb_zero_point;
      vacc12 += va10 * vb02;
      vacc12 += va11 * vb12;
      vacc12 += va12 * vb22;
      vacc12 += va13 * vb32;
      vacc12 -= ((int32_t) vasum1) * vb_zero_point;
      vacc13 += va10 * vb03;
      vacc13 += va11 * vb13;
      vacc13 += va12 * vb23;
      vacc13 += va13 * vb33;
      vacc13 -= ((int32_t) vasum1) * vb_zero_point;
      vacc20 += va20 * vb00;
      vacc20 += va21 * vb10;
      vacc20 += va22 * vb20;
      vacc20 += va23 * vb30;
      vacc20 -= ((int32_t) vasum2) * vb_zero_point;
      vacc21 += va20 * vb01;
      vacc21 += va21 * vb11;
      vacc21 += va22 * vb21;
      vacc21 += va23 * vb31;
      vacc21 -= ((int32_t) vasum2) * vb_zero_point;
      vacc22 += va20 * vb02;
      vacc22 += va21 * vb12;
      vacc22 += va22 * vb22;
      vacc22 += va23 * vb32;
      vacc22 -= ((int32_t) vasum2) * vb_zero_point;
      vacc23 += va20 * vb03;
      vacc23 += va21 * vb13;
      vacc23 += va22 * vb23;
      vacc23 += va23 * vb33;
      vacc23 -= ((int32_t) vasum2) * vb_zero_point;
      vacc30 += va30 * vb00;
      vacc30 += va31 * vb10;
      vacc30 += va32 * vb20;
      vacc30 += va33 * vb30;
      vacc30 -= ((int32_t) vasum3) * vb_zero_point;
      vacc31 += va30 * vb01;
      vacc31 += va31 * vb11;
      vacc31 += va32 * vb21;
      vacc31 += va33 * vb31;
      vacc31 -= ((int32_t) vasum3) * vb_zero_point;
      vacc32 += va30 * vb02;
      vacc32 += va31 * vb12;
      vacc32 += va32 * vb22;
      vacc32 += va33 * vb32;
      vacc32 -= ((int32_t) vasum3) * vb_zero_point;
      vacc33 += va30 * vb03;
      vacc33 += va31 * vb13;
      vacc33 += va32 * vb23;
      vacc33 += va33 * vb33;
      vacc33 -= ((int32_t) vasum3) * vb_zero_point;
      vacc40 += va40 * vb00;
      vacc40 += va41 * vb10;
      vacc40 += va42 * vb20;
      vacc40 += va43 * vb30;
      vacc40 -= ((int32_t) vasum4) * vb_zero_point;
      vacc41 += va40 * vb01;
      vacc41 += va41 * vb11;
      vacc41 += va42 * vb21;
      vacc41 += va43 * vb31;
      vacc41 -= ((int32_t) vasum4) * vb_zero_point;
      vacc42 += va40 * vb02;
      vacc42 += va41 * vb12;
      vacc42 += va42 * vb22;
      vacc42 += va43 * vb32;
      vacc42 -= ((int32_t) vasum4) * vb_zero_point;
      vacc43 += va40 * vb03;
      vacc43 += va41 * vb13;
      vacc43 += va42 * vb23;
      vacc43 += va43 * vb33;
      vacc43 -= ((int32_t) vasum4) * vb_zero_point;
      vacc50 += va50 * vb00;
      vacc50 += va51 * vb10;
      vacc50 += va52 * vb20;
      vacc50 += va53 * vb30;
      vacc50 -= ((int32_t) vasum5) * vb_zero_point;
      vacc51 += va50 * vb01;
      vacc51 += va51 * vb11;
      vacc51 += va52 * vb21;
      vacc51 += va53 * vb31;
      vacc51 -= ((int32_t) vasum5) * vb_zero_point;
      vacc52 += va50 * vb02;
      vacc52 += va51 * vb12;
      vacc52 += va52 * vb22;
      vacc52 += va53 * vb32;
      vacc52 -= ((int32_t) vasum5) * vb_zero_point;
      vacc53 += va50 * vb03;
      vacc53 += va51 * vb13;
      vacc53 += va52 * vb23;
      vacc53 += va53 * vb33;
      vacc53 -= ((int32_t) vasum5) * vb_zero_point;
      vacc60 += va60 * vb00;
      vacc60 += va61 * vb10;
      vacc60 += va62 * vb20;
      vacc60 += va63 * vb30;
      vacc60 -= ((int32_t) vasum6) * vb_zero_point;
      vacc61 += va60 * vb01;
      vacc61 += va61 * vb11;
      vacc61 += va62 * vb21;
      vacc61 += va63 * vb31;
      vacc61 -= ((int32_t) vasum6) * vb_zero_point;
      vacc62 += va60 * vb02;
      vacc62 += va61 * vb12;
      vacc62 += va62 * vb22;
      vacc62 += va63 * vb32;
      vacc62 -= ((int32_t) vasum6) * vb_zero_point;
      vacc63 += va60 * vb03;
      vacc63 += va61 * vb13;
      vacc63 += va62 * vb23;
      vacc63 += va63 * vb33;
      vacc63 -= ((int32_t) vasum6) * vb_zero_point;
      vacc70 += va70 * vb00;
      vacc70 += va71 * vb10;
      vacc70 += va72 * vb20;
      vacc70 += va73 * vb30;
      vacc70 -= ((int32_t) vasum7) * vb_zero_point;
      vacc71 += va70 * vb01;
      vacc71 += va71 * vb11;
      vacc71 += va72 * vb21;
      vacc71 += va73 * vb31;
      vacc71 -= ((int32_t) vasum7) * vb_zero_point;
      vacc72 += va70 * vb02;
      vacc72 += va71 * vb12;
      vacc72 += va72 * vb22;
      vacc72 += va73 * vb32;
      vacc72 -= ((int32_t) vasum7) * vb_zero_point;
      vacc73 += va70 * vb03;
      vacc73 += va71 * vb13;
      vacc73 += va72 * vb23;
      vacc73 += va73 * vb33;
      vacc73 -= ((int32_t) vasum7) * vb_zero_point;
      vacc80 += va80 * vb00;
      vacc80 += va81 * vb10;
      vacc80 += va82 * vb20;
      vacc80 += va83 * vb30;
      vacc80 -= ((int32_t) vasum8) * vb_zero_point;
      vacc81 += va80 * vb01;
      vacc81 += va81 * vb11;
      vacc81 += va82 * vb21;
      vacc81 += va83 * vb31;
      vacc81 -= ((int32_t) vasum8) * vb_zero_point;
      vacc82 += va80 * vb02;
      vacc82 += va81 * vb12;
      vacc82 += va82 * vb22;
      vacc82 += va83 * vb32;
      vacc82 -= ((int32_t) vasum8) * vb_zero_point;
      vacc83 += va80 * vb03;
      vacc83 += va81 * vb13;
      vacc83 += va82 * vb23;
      vacc83 += va83 * vb33;
      vacc83 -= ((int32_t) vasum8) * vb_zero_point;
      vacc90 += va90 * vb00;
      vacc90 += va91 * vb10;
      vacc90 += va92 * vb20;
      vacc90 += va93 * vb30;
      vacc90 -= ((int32_t) vasum9) * vb_zero_point;
      vacc91 += va90 * vb01;
      vacc91 += va91 * vb11;
      vacc91 += va92 * vb21;
      vacc91 += va93 * vb31;
      vacc91 -= ((int32_t) vasum9) * vb_zero_point;
      vacc92 += va90 * vb02;
      vacc92 += va91 * vb12;
      vacc92 += va92 * vb22;
      vacc92 += va93 * vb32;
      vacc92 -= ((int32_t) vasum9) * vb_zero_point;
      vacc93 += va90 * vb03;
      vacc93 += va91 * vb13;
      vacc93 += va92 * vb23;
      vacc93 += va93 * vb33;
      vacc93 -= ((int32_t) vasum9) * vb_zero_point;
      vacc100 += va100 * vb00;
      vacc100 += va101 * vb10;
      vacc100 += va102 * vb20;
      vacc100 += va103 * vb30;
      vacc100 -= ((int32_t) vasum10) * vb_zero_point;
      vacc101 += va100 * vb01;
      vacc101 += va101 * vb11;
      vacc101 += va102 * vb21;
      vacc101 += va103 * vb31;
      vacc101 -= ((int32_t) vasum10) * vb_zero_point;
      vacc102 += va100 * vb02;
      vacc102 += va101 * vb12;
      vacc102 += va102 * vb22;
      vacc102 += va103 * vb32;
      vacc102 -= ((int32_t) vasum10) * vb_zero_point;
      vacc103 += va100 * vb03;
      vacc103 += va101 * vb13;
      vacc103 += va102 * vb23;
      vacc103 += va103 * vb33;
      vacc103 -= ((int32_t) vasum10) * vb_zero_point;
      vacc110 += va110 * vb00;
      vacc110 += va111 * vb10;
      vacc110 += va112 * vb20;
      vacc110 += va113 * vb30;
      vacc110 -= ((int32_t) vasum11) * vb_zero_point;
      vacc111 += va110 * vb01;
      vacc111 += va111 * vb11;
      vacc111 += va112 * vb21;
      vacc111 += va113 * vb31;
      vacc111 -= ((int32_t) vasum11) * vb_zero_point;
      vacc112 += va110 * vb02;
      vacc112 += va111 * vb12;
      vacc112 += va112 * vb22;
      vacc112 += va113 * vb32;
      vacc112 -= ((int32_t) vasum11) * vb_zero_point;
      vacc113 += va110 * vb03;
      vacc113 += va111 * vb13;
      vacc113 += va112 * vb23;
      vacc113 += va113 * vb33;
      vacc113 -= ((int32_t) vasum11) * vb_zero_point;

      k += 4 * sizeof(uint8_t);
    }
    // End of accumulation loop. The variable `k` contains the amount by which
    // we advanced the `va` pointers, so we rewind by this amount now.
    a0 = (const uint8_t*)((uintptr_t)a0 - k);
    a1 = (const uint8_t*)((uintptr_t)a1 - k);
    a2 = (const uint8_t*)((uintptr_t)a2 - k);
    a3 = (const uint8_t*)((uintptr_t)a3 - k);
    a4 = (const uint8_t*)((uintptr_t)a4 - k);
    a5 = (const uint8_t*)((uintptr_t)a5 - k);
    a6 = (const uint8_t*)((uintptr_t)a6 - k);
    a7 = (const uint8_t*)((uintptr_t)a7 - k);
    a8 = (const uint8_t*)((uintptr_t)a8 - k);
    a9 = (const uint8_t*)((uintptr_t)a9 - k);
    a10 = (const uint8_t*)((uintptr_t)a10 - k);
    a11 = (const uint8_t*)((uintptr_t)a11 - k);

    // Post-accumulation work

    const int32_t vmultiplier = params->scalar.multiplier;
    const int64_t vq31rounding = INT64_C(0x40000000);
    const int32_t vremainder_mask = params->scalar.remainder_mask;
    const uint32_t vshift = params->scalar.shift;
    const int32_t vremainder_threshold = params->scalar.remainder_threshold;
    const int32_t vout_min = params->scalar.output_min_less_zero_point;
    const int32_t vout_max = params->scalar.output_max_less_zero_point;
    const int32_t voutput_zero_point = params->scalar.output_zero_point;

    // voutMN will hold the output value at row mi, column ni, before
    // we actually check if this is in bounds for the destination matrix.
    uint8_t vout00;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc00 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout00 = (uint8_t)out;
    }
    uint8_t vout01;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc01 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout01 = (uint8_t)out;
    }
    uint8_t vout02;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc02 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout02 = (uint8_t)out;
    }
    uint8_t vout03;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc03 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout03 = (uint8_t)out;
    }
    uint8_t vout10;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc10 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout10 = (uint8_t)out;
    }
    uint8_t vout11;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc11 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout11 = (uint8_t)out;
    }
    uint8_t vout12;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc12 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout12 = (uint8_t)out;
    }
    uint8_t vout13;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc13 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout13 = (uint8_t)out;
    }
    uint8_t vout20;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc20 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout20 = (uint8_t)out;
    }
    uint8_t vout21;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc21 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout21 = (uint8_t)out;
    }
    uint8_t vout22;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc22 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout22 = (uint8_t)out;
    }
    uint8_t vout23;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc23 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout23 = (uint8_t)out;
    }
    uint8_t vout30;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc30 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout30 = (uint8_t)out;
    }
    uint8_t vout31;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc31 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout31 = (uint8_t)out;
    }
    uint8_t vout32;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc32 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout32 = (uint8_t)out;
    }
    uint8_t vout33;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc33 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout33 = (uint8_t)out;
    }
    uint8_t vout40;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc40 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout40 = (uint8_t)out;
    }
    uint8_t vout41;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc41 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout41 = (uint8_t)out;
    }
    uint8_t vout42;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc42 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout42 = (uint8_t)out;
    }
    uint8_t vout43;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc43 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout43 = (uint8_t)out;
    }
    uint8_t vout50;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc50 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout50 = (uint8_t)out;
    }
    uint8_t vout51;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc51 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout51 = (uint8_t)out;
    }
    uint8_t vout52;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc52 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout52 = (uint8_t)out;
    }
    uint8_t vout53;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc53 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout53 = (uint8_t)out;
    }
    uint8_t vout60;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc60 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout60 = (uint8_t)out;
    }
    uint8_t vout61;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc61 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout61 = (uint8_t)out;
    }
    uint8_t vout62;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc62 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout62 = (uint8_t)out;
    }
    uint8_t vout63;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc63 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout63 = (uint8_t)out;
    }
    uint8_t vout70;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc70 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout70 = (uint8_t)out;
    }
    uint8_t vout71;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc71 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout71 = (uint8_t)out;
    }
    uint8_t vout72;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc72 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout72 = (uint8_t)out;
    }
    uint8_t vout73;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc73 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout73 = (uint8_t)out;
    }
    uint8_t vout80;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc80 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout80 = (uint8_t)out;
    }
    uint8_t vout81;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc81 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout81 = (uint8_t)out;
    }
    uint8_t vout82;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc82 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout82 = (uint8_t)out;
    }
    uint8_t vout83;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc83 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout83 = (uint8_t)out;
    }
    uint8_t vout90;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc90 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout90 = (uint8_t)out;
    }
    uint8_t vout91;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc91 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout91 = (uint8_t)out;
    }
    uint8_t vout92;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc92 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout92 = (uint8_t)out;
    }
    uint8_t vout93;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc93 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout93 = (uint8_t)out;
    }
    uint8_t vout100;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc100 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout100 = (uint8_t)out;
    }
    uint8_t vout101;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc101 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout101 = (uint8_t)out;
    }
    uint8_t vout102;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc102 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout102 = (uint8_t)out;
    }
    uint8_t vout103;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc103 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout103 = (uint8_t)out;
    }
    uint8_t vout110;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc110 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout110 = (uint8_t)out;
    }
    uint8_t vout111;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc111 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout111 = (uint8_t)out;
    }
    uint8_t vout112;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc112 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout112 = (uint8_t)out;
    }
    uint8_t vout113;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc113 * (int64_t)vmultiplier;
      const int32_t vq31product =
          (int32_t)(uint32_t)((uint64_t)(vproduct + vq31rounding) >> 31);
      const int32_t vremainder =
          (vq31product & vremainder_mask) - (int32_t)(vq31product < 0);
      int32_t out = asr_s32(vq31product, vshift) +
                    (int32_t)(vremainder > vremainder_threshold);
      // Apply the min-max clamp and add the output zero point.
      out = out < vout_min ? vout_min : out;
      out = out > vout_max ? vout_max : out;
      out += voutput_zero_point;
      vout113 = (uint8_t)out;
    }

    if XNN_LIKELY (nc >= 4) {
      // Main case where there the 4 columns fit in the destination.
      c0[0] = vout00;
      c0[1] = vout01;
      c0[2] = vout02;
      c0[3] = vout03;
      c1[0] = vout10;
      c1[1] = vout11;
      c1[2] = vout12;
      c1[3] = vout13;
      c2[0] = vout20;
      c2[1] = vout21;
      c2[2] = vout22;
      c2[3] = vout23;
      c3[0] = vout30;
      c3[1] = vout31;
      c3[2] = vout32;
      c3[3] = vout33;
      c4[0] = vout40;
      c4[1] = vout41;
      c4[2] = vout42;
      c4[3] = vout43;
      c5[0] = vout50;
      c5[1] = vout51;
      c5[2] = vout52;
      c5[3] = vout53;
      c6[0] = vout60;
      c6[1] = vout61;
      c6[2] = vout62;
      c6[3] = vout63;
      c7[0] = vout70;
      c7[1] = vout71;
      c7[2] = vout72;
      c7[3] = vout73;
      c8[0] = vout80;
      c8[1] = vout81;
      c8[2] = vout82;
      c8[3] = vout83;
      c9[0] = vout90;
      c9[1] = vout91;
      c9[2] = vout92;
      c9[3] = vout93;
      c10[0] = vout100;
      c10[1] = vout101;
      c10[2] = vout102;
      c10[3] = vout103;
      c11[0] = vout110;
      c11[1] = vout111;
      c11[2] = vout112;
      c11[3] = vout113;

      // Advance to the next 4 columns.
      c0 = (uint8_t*)((uintptr_t)c0 + cn_stride);
      c1 = (uint8_t*)((uintptr_t)c1 + cn_stride);
      c2 = (uint8_t*)((uintptr_t)c2 + cn_stride);
      c3 = (uint8_t*)((uintptr_t)c3 + cn_stride);
      c4 = (uint8_t*)((uintptr_t)c4 + cn_stride);
      c5 = (uint8_t*)((uintptr_t)c5 + cn_stride);
      c6 = (uint8_t*)((uintptr_t)c6 + cn_stride);
      c7 = (uint8_t*)((uintptr_t)c7 + cn_stride);
      c8 = (uint8_t*)((uintptr_t)c8 + cn_stride);
      c9 = (uint8_t*)((uintptr_t)c9 + cn_stride);
      c10 = (uint8_t*)((uintptr_t)c10 + cn_stride);
      c11 = (uint8_t*)((uintptr_t)c11 + cn_stride);

      nc -= 4;
    } else {
      // Final case where not all of the 4 columns fit in the destination.
      if (nc > 0) {
        c0[0] = vout00;
      }
      if (nc > 1) {
        c0[1] = vout01;
      }
      if (nc > 2) {
        c0[2] = vout02;
      }
      if (nc > 3) {
        c0[3] = vout03;
      }
      if (nc > 0) {
        c1[0] = vout10;
      }
      if (nc > 1) {
        c1[1] = vout11;
      }
      if (nc > 2) {
        c1[2] = vout12;
      }
      if (nc > 3) {
        c1[3] = vout13;
      }
      if (nc > 0) {
        c2[0] = vout20;
      }
      if (nc > 1) {
        c2[1] = vout21;
      }
      if (nc > 2) {
        c2[2] = vout22;
      }
      if (nc > 3) {
        c2[3] = vout23;
      }
      if (nc > 0) {
        c3[0] = vout30;
      }
      if (nc > 1) {
        c3[1] = vout31;
      }
      if (nc > 2) {
        c3[2] = vout32;
      }
      if (nc > 3) {
        c3[3] = vout33;
      }
      if (nc > 0) {
        c4[0] = vout40;
      }
      if (nc > 1) {
        c4[1] = vout41;
      }
      if (nc > 2) {
        c4[2] = vout42;
      }
      if (nc > 3) {
        c4[3] = vout43;
      }
      if (nc > 0) {
        c5[0] = vout50;
      }
      if (nc > 1) {
        c5[1] = vout51;
      }
      if (nc > 2) {
        c5[2] = vout52;
      }
      if (nc > 3) {
        c5[3] = vout53;
      }
      if (nc > 0) {
        c6[0] = vout60;
      }
      if (nc > 1) {
        c6[1] = vout61;
      }
      if (nc > 2) {
        c6[2] = vout62;
      }
      if (nc > 3) {
        c6[3] = vout63;
      }
      if (nc > 0) {
        c7[0] = vout70;
      }
      if (nc > 1) {
        c7[1] = vout71;
      }
      if (nc > 2) {
        c7[2] = vout72;
      }
      if (nc > 3) {
        c7[3] = vout73;
      }
      if (nc > 0) {
        c8[0] = vout80;
      }
      if (nc > 1) {
        c8[1] = vout81;
      }
      if (nc > 2) {
        c8[2] = vout82;
      }
      if (nc > 3) {
        c8[3] = vout83;
      }
      if (nc > 0) {
        c9[0] = vout90;
      }
      if (nc > 1) {
        c9[1] = vout91;
      }
      if (nc > 2) {
        c9[2] = vout92;
      }
      if (nc > 3) {
        c9[3] = vout93;
      }
      if (nc > 0) {
        c10[0] = vout100;
      }
      if (nc > 1) {
        c10[1] = vout101;
      }
      if (nc > 2) {
        c10[2] = vout102;
      }
      if (nc > 3) {
        c10[3] = vout103;
      }
      if (nc > 0) {
        c11[0] = vout110;
      }
      if (nc > 1) {
        c11[1] = vout111;
      }
      if (nc > 2) {
        c11[2] = vout112;
      }
      if (nc > 3) {
        c11[3] = vout113;
      }

      nc = 0;
    }
  } while (nc != 0);
}
