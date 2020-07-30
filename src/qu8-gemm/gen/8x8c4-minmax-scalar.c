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
void xnn_qu8_gemm_minmax_ukernel_8x8c4__scalar(
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
  assert(mr <= 8);
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
  if XNN_UNPREDICTABLE(mr != 8) {
    a7 = a6;
    c7 = c6;
  }

  const int32_t vb_zero_point = params->scalar.kernel_zero_point;

  // Loop over groups of 8 columns.
  do {
    // `vaccMN` is the accumulator at row `M`, column `N`.
    // Initialize accumulators with bias. 8 bias values are loaded from the
    // weight matrix, at the start of the group of 8 columns.
    int32_t bias0 = ((const int32_t*)w)[0];
    int32_t vacc00 = bias0;
    int32_t vacc10 = bias0;
    int32_t vacc20 = bias0;
    int32_t vacc30 = bias0;
    int32_t vacc40 = bias0;
    int32_t vacc50 = bias0;
    int32_t vacc60 = bias0;
    int32_t vacc70 = bias0;
    int32_t bias1 = ((const int32_t*)w)[1];
    int32_t vacc01 = bias1;
    int32_t vacc11 = bias1;
    int32_t vacc21 = bias1;
    int32_t vacc31 = bias1;
    int32_t vacc41 = bias1;
    int32_t vacc51 = bias1;
    int32_t vacc61 = bias1;
    int32_t vacc71 = bias1;
    int32_t bias2 = ((const int32_t*)w)[2];
    int32_t vacc02 = bias2;
    int32_t vacc12 = bias2;
    int32_t vacc22 = bias2;
    int32_t vacc32 = bias2;
    int32_t vacc42 = bias2;
    int32_t vacc52 = bias2;
    int32_t vacc62 = bias2;
    int32_t vacc72 = bias2;
    int32_t bias3 = ((const int32_t*)w)[3];
    int32_t vacc03 = bias3;
    int32_t vacc13 = bias3;
    int32_t vacc23 = bias3;
    int32_t vacc33 = bias3;
    int32_t vacc43 = bias3;
    int32_t vacc53 = bias3;
    int32_t vacc63 = bias3;
    int32_t vacc73 = bias3;
    int32_t bias4 = ((const int32_t*)w)[4];
    int32_t vacc04 = bias4;
    int32_t vacc14 = bias4;
    int32_t vacc24 = bias4;
    int32_t vacc34 = bias4;
    int32_t vacc44 = bias4;
    int32_t vacc54 = bias4;
    int32_t vacc64 = bias4;
    int32_t vacc74 = bias4;
    int32_t bias5 = ((const int32_t*)w)[5];
    int32_t vacc05 = bias5;
    int32_t vacc15 = bias5;
    int32_t vacc25 = bias5;
    int32_t vacc35 = bias5;
    int32_t vacc45 = bias5;
    int32_t vacc55 = bias5;
    int32_t vacc65 = bias5;
    int32_t vacc75 = bias5;
    int32_t bias6 = ((const int32_t*)w)[6];
    int32_t vacc06 = bias6;
    int32_t vacc16 = bias6;
    int32_t vacc26 = bias6;
    int32_t vacc36 = bias6;
    int32_t vacc46 = bias6;
    int32_t vacc56 = bias6;
    int32_t vacc66 = bias6;
    int32_t vacc76 = bias6;
    int32_t bias7 = ((const int32_t*)w)[7];
    int32_t vacc07 = bias7;
    int32_t vacc17 = bias7;
    int32_t vacc27 = bias7;
    int32_t vacc37 = bias7;
    int32_t vacc47 = bias7;
    int32_t vacc57 = bias7;
    int32_t vacc67 = bias7;
    int32_t vacc77 = bias7;

    w = (const void*)((uintptr_t)w + 8 * sizeof(int32_t));

    // Inner accumulation loop along the 8 columns.
    // Handle 4 rows at each iteration: this is key to modelling what an
    // actual kernel using ARMv8.2 dot-product instructions would look like.
    size_t k = 0;
    while (k < kc) {
      // Load a 8x4 block of activations, and compute sums along rows.
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

      // Load a 4x8 block of weights.
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
      uint8_t vb04 = ((const uint8_t*)w)[0];
      uint8_t vb14 = ((const uint8_t*)w)[1];
      uint8_t vb24 = ((const uint8_t*)w)[2];
      uint8_t vb34 = ((const uint8_t*)w)[3];

      w = (const void*)((uintptr_t)w + 4 * sizeof(uint8_t));
      uint8_t vb05 = ((const uint8_t*)w)[0];
      uint8_t vb15 = ((const uint8_t*)w)[1];
      uint8_t vb25 = ((const uint8_t*)w)[2];
      uint8_t vb35 = ((const uint8_t*)w)[3];

      w = (const void*)((uintptr_t)w + 4 * sizeof(uint8_t));
      uint8_t vb06 = ((const uint8_t*)w)[0];
      uint8_t vb16 = ((const uint8_t*)w)[1];
      uint8_t vb26 = ((const uint8_t*)w)[2];
      uint8_t vb36 = ((const uint8_t*)w)[3];

      w = (const void*)((uintptr_t)w + 4 * sizeof(uint8_t));
      uint8_t vb07 = ((const uint8_t*)w)[0];
      uint8_t vb17 = ((const uint8_t*)w)[1];
      uint8_t vb27 = ((const uint8_t*)w)[2];
      uint8_t vb37 = ((const uint8_t*)w)[3];

      w = (const void*)((uintptr_t)w + 4 * sizeof(uint8_t));

      // Multiply-accumulate: 8x4 * 4x8 --> 8x8. The inner size 4 here means
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
      vacc04 += va00 * vb04;
      vacc04 += va01 * vb14;
      vacc04 += va02 * vb24;
      vacc04 += va03 * vb34;
      vacc04 -= ((int32_t) vasum0) * vb_zero_point;
      vacc05 += va00 * vb05;
      vacc05 += va01 * vb15;
      vacc05 += va02 * vb25;
      vacc05 += va03 * vb35;
      vacc05 -= ((int32_t) vasum0) * vb_zero_point;
      vacc06 += va00 * vb06;
      vacc06 += va01 * vb16;
      vacc06 += va02 * vb26;
      vacc06 += va03 * vb36;
      vacc06 -= ((int32_t) vasum0) * vb_zero_point;
      vacc07 += va00 * vb07;
      vacc07 += va01 * vb17;
      vacc07 += va02 * vb27;
      vacc07 += va03 * vb37;
      vacc07 -= ((int32_t) vasum0) * vb_zero_point;
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
      vacc14 += va10 * vb04;
      vacc14 += va11 * vb14;
      vacc14 += va12 * vb24;
      vacc14 += va13 * vb34;
      vacc14 -= ((int32_t) vasum1) * vb_zero_point;
      vacc15 += va10 * vb05;
      vacc15 += va11 * vb15;
      vacc15 += va12 * vb25;
      vacc15 += va13 * vb35;
      vacc15 -= ((int32_t) vasum1) * vb_zero_point;
      vacc16 += va10 * vb06;
      vacc16 += va11 * vb16;
      vacc16 += va12 * vb26;
      vacc16 += va13 * vb36;
      vacc16 -= ((int32_t) vasum1) * vb_zero_point;
      vacc17 += va10 * vb07;
      vacc17 += va11 * vb17;
      vacc17 += va12 * vb27;
      vacc17 += va13 * vb37;
      vacc17 -= ((int32_t) vasum1) * vb_zero_point;
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
      vacc24 += va20 * vb04;
      vacc24 += va21 * vb14;
      vacc24 += va22 * vb24;
      vacc24 += va23 * vb34;
      vacc24 -= ((int32_t) vasum2) * vb_zero_point;
      vacc25 += va20 * vb05;
      vacc25 += va21 * vb15;
      vacc25 += va22 * vb25;
      vacc25 += va23 * vb35;
      vacc25 -= ((int32_t) vasum2) * vb_zero_point;
      vacc26 += va20 * vb06;
      vacc26 += va21 * vb16;
      vacc26 += va22 * vb26;
      vacc26 += va23 * vb36;
      vacc26 -= ((int32_t) vasum2) * vb_zero_point;
      vacc27 += va20 * vb07;
      vacc27 += va21 * vb17;
      vacc27 += va22 * vb27;
      vacc27 += va23 * vb37;
      vacc27 -= ((int32_t) vasum2) * vb_zero_point;
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
      vacc34 += va30 * vb04;
      vacc34 += va31 * vb14;
      vacc34 += va32 * vb24;
      vacc34 += va33 * vb34;
      vacc34 -= ((int32_t) vasum3) * vb_zero_point;
      vacc35 += va30 * vb05;
      vacc35 += va31 * vb15;
      vacc35 += va32 * vb25;
      vacc35 += va33 * vb35;
      vacc35 -= ((int32_t) vasum3) * vb_zero_point;
      vacc36 += va30 * vb06;
      vacc36 += va31 * vb16;
      vacc36 += va32 * vb26;
      vacc36 += va33 * vb36;
      vacc36 -= ((int32_t) vasum3) * vb_zero_point;
      vacc37 += va30 * vb07;
      vacc37 += va31 * vb17;
      vacc37 += va32 * vb27;
      vacc37 += va33 * vb37;
      vacc37 -= ((int32_t) vasum3) * vb_zero_point;
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
      vacc44 += va40 * vb04;
      vacc44 += va41 * vb14;
      vacc44 += va42 * vb24;
      vacc44 += va43 * vb34;
      vacc44 -= ((int32_t) vasum4) * vb_zero_point;
      vacc45 += va40 * vb05;
      vacc45 += va41 * vb15;
      vacc45 += va42 * vb25;
      vacc45 += va43 * vb35;
      vacc45 -= ((int32_t) vasum4) * vb_zero_point;
      vacc46 += va40 * vb06;
      vacc46 += va41 * vb16;
      vacc46 += va42 * vb26;
      vacc46 += va43 * vb36;
      vacc46 -= ((int32_t) vasum4) * vb_zero_point;
      vacc47 += va40 * vb07;
      vacc47 += va41 * vb17;
      vacc47 += va42 * vb27;
      vacc47 += va43 * vb37;
      vacc47 -= ((int32_t) vasum4) * vb_zero_point;
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
      vacc54 += va50 * vb04;
      vacc54 += va51 * vb14;
      vacc54 += va52 * vb24;
      vacc54 += va53 * vb34;
      vacc54 -= ((int32_t) vasum5) * vb_zero_point;
      vacc55 += va50 * vb05;
      vacc55 += va51 * vb15;
      vacc55 += va52 * vb25;
      vacc55 += va53 * vb35;
      vacc55 -= ((int32_t) vasum5) * vb_zero_point;
      vacc56 += va50 * vb06;
      vacc56 += va51 * vb16;
      vacc56 += va52 * vb26;
      vacc56 += va53 * vb36;
      vacc56 -= ((int32_t) vasum5) * vb_zero_point;
      vacc57 += va50 * vb07;
      vacc57 += va51 * vb17;
      vacc57 += va52 * vb27;
      vacc57 += va53 * vb37;
      vacc57 -= ((int32_t) vasum5) * vb_zero_point;
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
      vacc64 += va60 * vb04;
      vacc64 += va61 * vb14;
      vacc64 += va62 * vb24;
      vacc64 += va63 * vb34;
      vacc64 -= ((int32_t) vasum6) * vb_zero_point;
      vacc65 += va60 * vb05;
      vacc65 += va61 * vb15;
      vacc65 += va62 * vb25;
      vacc65 += va63 * vb35;
      vacc65 -= ((int32_t) vasum6) * vb_zero_point;
      vacc66 += va60 * vb06;
      vacc66 += va61 * vb16;
      vacc66 += va62 * vb26;
      vacc66 += va63 * vb36;
      vacc66 -= ((int32_t) vasum6) * vb_zero_point;
      vacc67 += va60 * vb07;
      vacc67 += va61 * vb17;
      vacc67 += va62 * vb27;
      vacc67 += va63 * vb37;
      vacc67 -= ((int32_t) vasum6) * vb_zero_point;
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
      vacc74 += va70 * vb04;
      vacc74 += va71 * vb14;
      vacc74 += va72 * vb24;
      vacc74 += va73 * vb34;
      vacc74 -= ((int32_t) vasum7) * vb_zero_point;
      vacc75 += va70 * vb05;
      vacc75 += va71 * vb15;
      vacc75 += va72 * vb25;
      vacc75 += va73 * vb35;
      vacc75 -= ((int32_t) vasum7) * vb_zero_point;
      vacc76 += va70 * vb06;
      vacc76 += va71 * vb16;
      vacc76 += va72 * vb26;
      vacc76 += va73 * vb36;
      vacc76 -= ((int32_t) vasum7) * vb_zero_point;
      vacc77 += va70 * vb07;
      vacc77 += va71 * vb17;
      vacc77 += va72 * vb27;
      vacc77 += va73 * vb37;
      vacc77 -= ((int32_t) vasum7) * vb_zero_point;

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
    uint8_t vout04;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc04 * (int64_t)vmultiplier;
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
      vout04 = (uint8_t)out;
    }
    uint8_t vout05;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc05 * (int64_t)vmultiplier;
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
      vout05 = (uint8_t)out;
    }
    uint8_t vout06;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc06 * (int64_t)vmultiplier;
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
      vout06 = (uint8_t)out;
    }
    uint8_t vout07;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc07 * (int64_t)vmultiplier;
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
      vout07 = (uint8_t)out;
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
    uint8_t vout14;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc14 * (int64_t)vmultiplier;
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
      vout14 = (uint8_t)out;
    }
    uint8_t vout15;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc15 * (int64_t)vmultiplier;
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
      vout15 = (uint8_t)out;
    }
    uint8_t vout16;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc16 * (int64_t)vmultiplier;
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
      vout16 = (uint8_t)out;
    }
    uint8_t vout17;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc17 * (int64_t)vmultiplier;
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
      vout17 = (uint8_t)out;
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
    uint8_t vout24;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc24 * (int64_t)vmultiplier;
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
      vout24 = (uint8_t)out;
    }
    uint8_t vout25;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc25 * (int64_t)vmultiplier;
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
      vout25 = (uint8_t)out;
    }
    uint8_t vout26;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc26 * (int64_t)vmultiplier;
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
      vout26 = (uint8_t)out;
    }
    uint8_t vout27;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc27 * (int64_t)vmultiplier;
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
      vout27 = (uint8_t)out;
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
    uint8_t vout34;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc34 * (int64_t)vmultiplier;
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
      vout34 = (uint8_t)out;
    }
    uint8_t vout35;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc35 * (int64_t)vmultiplier;
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
      vout35 = (uint8_t)out;
    }
    uint8_t vout36;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc36 * (int64_t)vmultiplier;
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
      vout36 = (uint8_t)out;
    }
    uint8_t vout37;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc37 * (int64_t)vmultiplier;
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
      vout37 = (uint8_t)out;
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
    uint8_t vout44;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc44 * (int64_t)vmultiplier;
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
      vout44 = (uint8_t)out;
    }
    uint8_t vout45;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc45 * (int64_t)vmultiplier;
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
      vout45 = (uint8_t)out;
    }
    uint8_t vout46;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc46 * (int64_t)vmultiplier;
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
      vout46 = (uint8_t)out;
    }
    uint8_t vout47;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc47 * (int64_t)vmultiplier;
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
      vout47 = (uint8_t)out;
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
    uint8_t vout54;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc54 * (int64_t)vmultiplier;
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
      vout54 = (uint8_t)out;
    }
    uint8_t vout55;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc55 * (int64_t)vmultiplier;
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
      vout55 = (uint8_t)out;
    }
    uint8_t vout56;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc56 * (int64_t)vmultiplier;
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
      vout56 = (uint8_t)out;
    }
    uint8_t vout57;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc57 * (int64_t)vmultiplier;
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
      vout57 = (uint8_t)out;
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
    uint8_t vout64;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc64 * (int64_t)vmultiplier;
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
      vout64 = (uint8_t)out;
    }
    uint8_t vout65;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc65 * (int64_t)vmultiplier;
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
      vout65 = (uint8_t)out;
    }
    uint8_t vout66;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc66 * (int64_t)vmultiplier;
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
      vout66 = (uint8_t)out;
    }
    uint8_t vout67;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc67 * (int64_t)vmultiplier;
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
      vout67 = (uint8_t)out;
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
    uint8_t vout74;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc74 * (int64_t)vmultiplier;
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
      vout74 = (uint8_t)out;
    }
    uint8_t vout75;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc75 * (int64_t)vmultiplier;
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
      vout75 = (uint8_t)out;
    }
    uint8_t vout76;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc76 * (int64_t)vmultiplier;
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
      vout76 = (uint8_t)out;
    }
    uint8_t vout77;
    {
      // Apply the quantized multiplier
      const int64_t vproduct = (int64_t)vacc77 * (int64_t)vmultiplier;
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
      vout77 = (uint8_t)out;
    }

    if XNN_LIKELY (nc >= 8) {
      // Main case where there the 8 columns fit in the destination.
      c0[0] = vout00;
      c0[1] = vout01;
      c0[2] = vout02;
      c0[3] = vout03;
      c0[4] = vout04;
      c0[5] = vout05;
      c0[6] = vout06;
      c0[7] = vout07;
      c1[0] = vout10;
      c1[1] = vout11;
      c1[2] = vout12;
      c1[3] = vout13;
      c1[4] = vout14;
      c1[5] = vout15;
      c1[6] = vout16;
      c1[7] = vout17;
      c2[0] = vout20;
      c2[1] = vout21;
      c2[2] = vout22;
      c2[3] = vout23;
      c2[4] = vout24;
      c2[5] = vout25;
      c2[6] = vout26;
      c2[7] = vout27;
      c3[0] = vout30;
      c3[1] = vout31;
      c3[2] = vout32;
      c3[3] = vout33;
      c3[4] = vout34;
      c3[5] = vout35;
      c3[6] = vout36;
      c3[7] = vout37;
      c4[0] = vout40;
      c4[1] = vout41;
      c4[2] = vout42;
      c4[3] = vout43;
      c4[4] = vout44;
      c4[5] = vout45;
      c4[6] = vout46;
      c4[7] = vout47;
      c5[0] = vout50;
      c5[1] = vout51;
      c5[2] = vout52;
      c5[3] = vout53;
      c5[4] = vout54;
      c5[5] = vout55;
      c5[6] = vout56;
      c5[7] = vout57;
      c6[0] = vout60;
      c6[1] = vout61;
      c6[2] = vout62;
      c6[3] = vout63;
      c6[4] = vout64;
      c6[5] = vout65;
      c6[6] = vout66;
      c6[7] = vout67;
      c7[0] = vout70;
      c7[1] = vout71;
      c7[2] = vout72;
      c7[3] = vout73;
      c7[4] = vout74;
      c7[5] = vout75;
      c7[6] = vout76;
      c7[7] = vout77;

      // Advance to the next 8 columns.
      c0 = (uint8_t*)((uintptr_t)c0 + cn_stride);
      c1 = (uint8_t*)((uintptr_t)c1 + cn_stride);
      c2 = (uint8_t*)((uintptr_t)c2 + cn_stride);
      c3 = (uint8_t*)((uintptr_t)c3 + cn_stride);
      c4 = (uint8_t*)((uintptr_t)c4 + cn_stride);
      c5 = (uint8_t*)((uintptr_t)c5 + cn_stride);
      c6 = (uint8_t*)((uintptr_t)c6 + cn_stride);
      c7 = (uint8_t*)((uintptr_t)c7 + cn_stride);

      nc -= 8;
    } else {
      // Final case where not all of the 8 columns fit in the destination.
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
      if (nc > 4) {
        c0[4] = vout04;
      }
      if (nc > 5) {
        c0[5] = vout05;
      }
      if (nc > 6) {
        c0[6] = vout06;
      }
      if (nc > 7) {
        c0[7] = vout07;
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
      if (nc > 4) {
        c1[4] = vout14;
      }
      if (nc > 5) {
        c1[5] = vout15;
      }
      if (nc > 6) {
        c1[6] = vout16;
      }
      if (nc > 7) {
        c1[7] = vout17;
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
      if (nc > 4) {
        c2[4] = vout24;
      }
      if (nc > 5) {
        c2[5] = vout25;
      }
      if (nc > 6) {
        c2[6] = vout26;
      }
      if (nc > 7) {
        c2[7] = vout27;
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
      if (nc > 4) {
        c3[4] = vout34;
      }
      if (nc > 5) {
        c3[5] = vout35;
      }
      if (nc > 6) {
        c3[6] = vout36;
      }
      if (nc > 7) {
        c3[7] = vout37;
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
      if (nc > 4) {
        c4[4] = vout44;
      }
      if (nc > 5) {
        c4[5] = vout45;
      }
      if (nc > 6) {
        c4[6] = vout46;
      }
      if (nc > 7) {
        c4[7] = vout47;
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
      if (nc > 4) {
        c5[4] = vout54;
      }
      if (nc > 5) {
        c5[5] = vout55;
      }
      if (nc > 6) {
        c5[6] = vout56;
      }
      if (nc > 7) {
        c5[7] = vout57;
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
      if (nc > 4) {
        c6[4] = vout64;
      }
      if (nc > 5) {
        c6[5] = vout65;
      }
      if (nc > 6) {
        c6[6] = vout66;
      }
      if (nc > 7) {
        c6[7] = vout67;
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
      if (nc > 4) {
        c7[4] = vout74;
      }
      if (nc > 5) {
        c7[5] = vout75;
      }
      if (nc > 6) {
        c7[6] = vout76;
      }
      if (nc > 7) {
        c7[7] = vout77;
      }

      nc = 0;
    }
  } while (nc != 0);
}
