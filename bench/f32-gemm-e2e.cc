// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <xnnpack.h>

#include <benchmark/benchmark.h>

#include "bench/utils.h"
#include "models/models.h"
#include <xnnpack/gemm.h>
#include <xnnpack/igemm.h>
#include <xnnpack/params.h>


static void GEMMEnd2EndBenchmark(
  benchmark::State& state,
  models::ExecutionPlanFactory model_factory,
  xnn_f32_gemm_ukernel_function gemm,
  xnn_f32_igemm_ukernel_function igemm,
  xnn_f32_gemm_ukernel_function gemm1,
  xnn_f32_igemm_ukernel_function igemm1,
  uint8_t mr, uint8_t nr, uint8_t log2_kr = 0, uint8_t log2_sr = 0,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check && !isa_check(state)) {
    return;
  }
  if (xnn_initialize(nullptr /* allocator */) != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  // Override microkernels chosen in xnn_initialize
  xnn_params.f32.gemm = (struct gemm_parameters) {
    .gemm = xnn_gemm_ukernel_function(gemm),
    .igemm = xnn_igemm_ukernel_function(igemm),
    .gemm1 = xnn_gemm_ukernel_function(gemm1),
    .igemm1 = xnn_igemm_ukernel_function(igemm1),
    .mr = mr,
    .nr = nr,
    .log2_kr = log2_kr,
    .log2_sr = log2_sr,
  };

  auto execution_plan = model_factory(nullptr);
  if (execution_plan.empty()) {
    state.SkipWithError("failed to create a model");
    return;
  }

  for (auto _ : state) {
    for (const std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>& op : execution_plan) {
      xnn_status status = xnn_run_operator(op.get(), nullptr);
      if (status != xnn_status_success) {
        state.SkipWithError("failed to run a model");
        return;
      }
    }
  }
  state.counters["Freq"] = benchmark::utils::GetCurrentCpuFrequency();
}

#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void f32_gemm_4x12__aarch64_neonfma_cortex_a53(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_4x12__aarch64_neonfma_cortex_a53,
      xnn_f32_igemm_ukernel_4x12__aarch64_neonfma_cortex_a53,
      xnn_f32_gemm_ukernel_1x12__aarch64_neonfma_cortex_a53,
      xnn_f32_igemm_ukernel_1x12__aarch64_neonfma_cortex_a53,
      4 /* mr */, 12 /* nr */);
  }

  static void f32_gemm_4x8__aarch64_neonfma_cortex_a53(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a53,
      xnn_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53,
      xnn_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a53,
      xnn_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53,
      4 /* mr */, 8 /* nr */);
  }

  static void f32_gemm_4x8__aarch64_neonfma_cortex_a57(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a57,
      xnn_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75 /* no A57 version */,
      xnn_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a57,
      xnn_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a57,
      4 /* mr */, 8 /* nr */);
  }

  static void f32_gemm_4x8__aarch64_neonfma_cortex_a75(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a75,
      xnn_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75,
      xnn_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75,
      xnn_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75,
      4 /* mr */, 8 /* nr */);
  }

  static void f32_gemm_4x8__aarch64_neonfma_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_4x8__aarch64_neonfma_ld64,
      xnn_f32_igemm_ukernel_4x8__neonfma_lane_ld64,
      xnn_f32_gemm_ukernel_1x8__neonfma_lane_ld64,
      xnn_f32_igemm_ukernel_1x8__neonfma_lane_ld64,
      4 /* mr */, 8 /* nr */);
  }

  static void f32_gemm_4x8__aarch64_neonfma_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_4x8__aarch64_neonfma_ld128,
      xnn_f32_igemm_ukernel_4x8__neonfma_lane_ld128,
      xnn_f32_gemm_ukernel_1x8__neonfma_lane_ld64,
      xnn_f32_igemm_ukernel_1x8__neonfma_lane_ld64,
      4 /* mr */, 8 /* nr */);
  }

  static void f32_gemm_5x8__aarch64_neonfma_cortex_a75(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_5x8__aarch64_neonfma_cortex_a75,
      xnn_f32_igemm_ukernel_5x8__aarch64_neonfma_cortex_a75,
      xnn_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75,
      xnn_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75,
      5 /* mr */, 8 /* nr */);
  }

  static void f32_gemm_6x8__aarch64_neonfma_cortex_a53(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a53,
      xnn_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53,
      xnn_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a53,
      xnn_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53,
      6 /* mr */, 8 /* nr */);
  }

  static void f32_gemm_6x8__aarch64_neonfma_cortex_a57(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a57,
      xnn_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a57,
      xnn_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a57,
      xnn_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a57,
      6 /* mr */, 8 /* nr */);
  }

  static void f32_gemm_6x8__aarch64_neonfma_cortex_a73(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a73,
      xnn_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a73,
      xnn_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75,
      xnn_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75,
      6 /* mr */, 8 /* nr */);
  }

  static void f32_gemm_6x8__aarch64_neonfma_cortex_a75(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75,
      xnn_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75,
      xnn_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75,
      xnn_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75,
      6 /* mr */, 8 /* nr */);
  }

  static void f32_gemm_6x8__aarch64_neonfma_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_6x8__aarch64_neonfma_ld64,
      xnn_f32_igemm_ukernel_6x8__neonfma_lane_ld64,
      xnn_f32_gemm_ukernel_1x8__neonfma_lane_ld64,
      xnn_f32_igemm_ukernel_1x8__neonfma_lane_ld64,
      6 /* mr */, 8 /* nr */);
  }

  static void f32_gemm_6x8__aarch64_neonfma_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_6x8__aarch64_neonfma_ld128,
      xnn_f32_igemm_ukernel_6x8__neonfma_lane_ld64,
      xnn_f32_gemm_ukernel_1x8__neonfma_lane_ld64,
      xnn_f32_igemm_ukernel_1x8__neonfma_lane_ld64,
      6 /* mr */, 8 /* nr */);
  }

  static void f32_gemm_4x8__neonfma_lane_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_4x8__neonfma_lane_ld64,
      xnn_f32_igemm_ukernel_4x8__neonfma_lane_ld64,
      xnn_f32_gemm_ukernel_1x8__neonfma_lane_ld64,
      xnn_f32_igemm_ukernel_1x8__neonfma_lane_ld64,
      4 /* mr */, 8 /* nr */);
  }

  static void f32_gemm_4x8__neonfma_lane_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_4x8__neonfma_lane_ld128,
      xnn_f32_igemm_ukernel_4x8__neonfma_lane_ld128,
      xnn_f32_gemm_ukernel_1x8__neonfma_lane_ld64,
      xnn_f32_igemm_ukernel_1x8__neonfma_lane_ld64,
      4 /* mr */, 8 /* nr */);
  }

  static void f32_gemm_6x8__neonfma_lane_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_6x8__neonfma_lane_ld64,
      xnn_f32_igemm_ukernel_6x8__neonfma_lane_ld64,
      xnn_f32_gemm_ukernel_1x8__neonfma_lane_ld64,
      xnn_f32_igemm_ukernel_1x8__neonfma_lane_ld64,
      6 /* mr */, 8 /* nr */);
  }

  static void f32_gemm_6x8__neonfma_lane_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_6x8__neonfma_lane_ld128,
      xnn_f32_igemm_ukernel_6x8__neonfma_lane_ld128,
      xnn_f32_gemm_ukernel_1x8__neonfma_lane_ld64,
      xnn_f32_igemm_ukernel_1x8__neonfma_lane_ld64,
      6 /* mr */, 8 /* nr */);
  }

  BENCHMARK_CAPTURE(f32_gemm_4x12__aarch64_neonfma_cortex_a53, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_4x12__aarch64_neonfma_cortex_a53, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_4x8__aarch64_neonfma_cortex_a53, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_4x8__aarch64_neonfma_cortex_a53, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_4x8__aarch64_neonfma_cortex_a57, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_4x8__aarch64_neonfma_cortex_a57, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_4x8__aarch64_neonfma_cortex_a75, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_4x8__aarch64_neonfma_cortex_a75, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_4x8__aarch64_neonfma_ld64, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_4x8__aarch64_neonfma_ld64, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_4x8__aarch64_neonfma_ld128, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_4x8__aarch64_neonfma_ld128, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_5x8__aarch64_neonfma_cortex_a75, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_5x8__aarch64_neonfma_cortex_a75, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_6x8__aarch64_neonfma_cortex_a53, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_6x8__aarch64_neonfma_cortex_a53, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_6x8__aarch64_neonfma_cortex_a57, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_6x8__aarch64_neonfma_cortex_a57, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_6x8__aarch64_neonfma_cortex_a73, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_6x8__aarch64_neonfma_cortex_a73, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_6x8__aarch64_neonfma_cortex_a75, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_6x8__aarch64_neonfma_cortex_a75, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_6x8__aarch64_neonfma_ld64, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_6x8__aarch64_neonfma_ld64, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_6x8__aarch64_neonfma_ld128, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_6x8__aarch64_neonfma_ld128, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_4x8__neonfma_lane_ld64, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_4x8__neonfma_lane_ld64, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_4x8__neonfma_lane_ld128, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_4x8__neonfma_lane_ld128, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_6x8__neonfma_lane_ld64, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_6x8__neonfma_lane_ld64, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_6x8__neonfma_lane_ld128, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_6x8__neonfma_lane_ld128, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_gemm_4x8__neon_lane_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_4x8__neon_lane_ld64,
      xnn_f32_igemm_ukernel_4x8__neon_lane_ld64,
      xnn_f32_gemm_ukernel_1x8__neon_lane_ld64,
      xnn_f32_igemm_ukernel_1x8__neon_lane_ld64,
      4 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void f32_gemm_4x8__neon_lane_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_4x8__neon_lane_ld128,
      xnn_f32_igemm_ukernel_4x8__neon_lane_ld128,
      xnn_f32_gemm_ukernel_1x8__neon_lane_ld64,
      xnn_f32_igemm_ukernel_1x8__neon_lane_ld64,
      4 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void f32_gemm_6x8__neon_lane_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_6x8__neon_lane_ld64,
      xnn_f32_igemm_ukernel_6x8__neon_lane_ld64,
      xnn_f32_gemm_ukernel_1x8__neon_lane_ld64,
      xnn_f32_igemm_ukernel_1x8__neon_lane_ld64,
      6 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void f32_gemm_6x8__neon_lane_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_6x8__neon_lane_ld128,
      xnn_f32_igemm_ukernel_6x8__neon_lane_ld128,
      xnn_f32_gemm_ukernel_1x8__neon_lane_ld64,
      xnn_f32_igemm_ukernel_1x8__neon_lane_ld64,
      6 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void f32_gemm_4x8__neon_dup_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_4x8__neon_dup_ld64,
      xnn_f32_igemm_ukernel_4x8__neon_dup_ld64,
      xnn_f32_gemm_ukernel_1x8__neon_dup_ld64,
      xnn_f32_igemm_ukernel_1x8__neon_dup_ld64,
      4 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void f32_gemm_4x8__neon_dup_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_4x8__neon_dup_ld128,
      xnn_f32_igemm_ukernel_4x8__neon_dup_ld128,
      xnn_f32_gemm_ukernel_1x8__neon_dup_ld64,
      xnn_f32_igemm_ukernel_1x8__neon_dup_ld64,
      4 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void f32_gemm_6x8__neon_dup_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_6x8__neon_dup_ld64,
      xnn_f32_igemm_ukernel_6x8__neon_dup_ld64,
      xnn_f32_gemm_ukernel_1x8__neon_dup_ld64,
      xnn_f32_igemm_ukernel_1x8__neon_dup_ld64,
      6 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void f32_gemm_6x8__neon_dup_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_6x8__neon_dup_ld128,
      xnn_f32_igemm_ukernel_6x8__neon_dup_ld128,
      xnn_f32_gemm_ukernel_1x8__neon_dup_ld64,
      xnn_f32_igemm_ukernel_1x8__neon_dup_ld64,
      6 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void f32_gemm_4x8__neonfma_dup_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_4x8__neonfma_dup_ld64,
      xnn_f32_igemm_ukernel_4x8__neonfma_dup_ld64,
      xnn_f32_gemm_ukernel_1x8__neonfma_dup_ld64,
      xnn_f32_igemm_ukernel_1x8__neonfma_dup_ld64,
      4 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONFMA);
  }

  static void f32_gemm_4x8__neonfma_dup_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_4x8__neonfma_dup_ld128,
      xnn_f32_igemm_ukernel_4x8__neonfma_dup_ld128,
      xnn_f32_gemm_ukernel_1x8__neonfma_dup_ld64,
      xnn_f32_igemm_ukernel_1x8__neonfma_dup_ld64,
      4 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONFMA);
  }

  static void f32_gemm_6x8__neonfma_dup_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_6x8__neonfma_dup_ld64,
      xnn_f32_igemm_ukernel_6x8__neonfma_dup_ld64,
      xnn_f32_gemm_ukernel_1x8__neonfma_dup_ld64,
      xnn_f32_igemm_ukernel_1x8__neonfma_dup_ld64,
      6 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONFMA);
  }

  static void f32_gemm_6x8__neonfma_dup_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_6x8__neonfma_dup_ld128,
      xnn_f32_igemm_ukernel_6x8__neonfma_dup_ld128,
      xnn_f32_gemm_ukernel_1x8__neonfma_dup_ld64,
      xnn_f32_igemm_ukernel_1x8__neonfma_dup_ld64,
      6 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONFMA);
  }

  static void f32_gemm_4x8s4__neon(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_4x8s4__neon,
      xnn_f32_igemm_ukernel_4x8s4__neon,
      xnn_f32_gemm_ukernel_1x8s4__neon,
      xnn_f32_igemm_ukernel_1x8s4__neon,
      4 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */,
      benchmark::utils::CheckNEON);
  }

  static void f32_gemm_4x8s4__neonfma(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_4x8s4__neonfma,
      xnn_f32_igemm_ukernel_4x8s4__neonfma,
      xnn_f32_gemm_ukernel_1x8s4__neonfma,
      xnn_f32_igemm_ukernel_1x8s4__neonfma,
      4 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */,
      benchmark::utils::CheckNEONFMA);
  }

  static void f32_gemm_6x8s4__neon(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_6x8s4__neon,
      xnn_f32_igemm_ukernel_6x8s4__neon,
      xnn_f32_gemm_ukernel_1x8s4__neon,
      xnn_f32_igemm_ukernel_1x8s4__neon,
      6 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */,
      benchmark::utils::CheckNEON);
  }

  static void f32_gemm_6x8s4__neonfma(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_6x8s4__neonfma,
      xnn_f32_igemm_ukernel_6x8s4__neonfma,
      xnn_f32_gemm_ukernel_1x8s4__neonfma,
      xnn_f32_igemm_ukernel_1x8s4__neonfma,
      6 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */,
      benchmark::utils::CheckNEONFMA);
  }

  static void f32_gemm_8x8s4__neon(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_8x8s4__neon,
      xnn_f32_igemm_ukernel_8x8s4__neon,
      xnn_f32_gemm_ukernel_1x8s4__neon,
      xnn_f32_igemm_ukernel_1x8s4__neon,
      8 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */,
      benchmark::utils::CheckNEON);
  }

  static void f32_gemm_8x8s4__neonfma(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_8x8s4__neonfma,
      xnn_f32_igemm_ukernel_8x8s4__neonfma,
      xnn_f32_gemm_ukernel_1x8s4__neonfma,
      xnn_f32_igemm_ukernel_1x8s4__neonfma,
      8 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_CAPTURE(f32_gemm_4x8__neon_lane_ld64, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_4x8__neon_lane_ld64, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_4x8__neon_lane_ld128, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_4x8__neon_lane_ld128, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_6x8__neon_lane_ld64, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_6x8__neon_lane_ld64, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_6x8__neon_lane_ld128, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_6x8__neon_lane_ld128, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_4x8__neon_dup_ld64, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_4x8__neon_dup_ld64, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_4x8__neon_dup_ld128, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_4x8__neon_dup_ld128, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_6x8__neon_dup_ld64, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_6x8__neon_dup_ld64, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_6x8__neon_dup_ld128, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_6x8__neon_dup_ld128, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_4x8__neonfma_dup_ld64, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_4x8__neonfma_dup_ld64, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_4x8__neonfma_dup_ld128, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_4x8__neonfma_dup_ld128, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_6x8__neonfma_dup_ld64, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_6x8__neonfma_dup_ld64, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_6x8__neonfma_dup_ld128, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_6x8__neonfma_dup_ld128, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_4x8s4__neon, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_4x8s4__neon, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_4x8s4__neonfma, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_4x8s4__neonfma, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_6x8s4__neon, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_6x8s4__neon, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_6x8s4__neonfma, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_6x8s4__neonfma, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_8x8s4__neon, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_8x8s4__neon, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_8x8s4__neonfma, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_8x8s4__neonfma, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void f32_gemm_4x8__sse_load1(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_4x8__sse_load1,
      xnn_f32_igemm_ukernel_4x8__sse_load1,
      xnn_f32_gemm_ukernel_1x8__sse_load1,
      xnn_f32_igemm_ukernel_1x8__sse_load1,
      4 /* mr */, 8 /* nr */);
  }

  static void f32_gemm_4x8__sse_dup(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_4x8__sse_dup,
      xnn_f32_igemm_ukernel_4x8__sse_dup,
      xnn_f32_gemm_ukernel_1x8__sse_dup,
      xnn_f32_igemm_ukernel_1x8__sse_dup,
      4 /* mr */, 8 /* nr */);
  }

  static void f32_gemm_4x8s4__sse(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_4x8s4__sse,
      xnn_f32_igemm_ukernel_4x8s4__sse,
      xnn_f32_gemm_ukernel_1x8s4__sse,
      xnn_f32_igemm_ukernel_1x8s4__sse,
      4 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */);
  }

  static void f32_gemm_4x8__avx_broadcast(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_4x8__avx_broadcast,
      xnn_f32_igemm_ukernel_4x8__avx_broadcast,
      xnn_f32_gemm_ukernel_1x8__avx_broadcast,
      xnn_f32_igemm_ukernel_1x8__avx_broadcast,
      4 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX);
  }

  static void f32_gemm_5x8__avx_broadcast(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_5x8__avx_broadcast,
      xnn_f32_igemm_ukernel_5x8__avx_broadcast,
      xnn_f32_gemm_ukernel_1x8__avx_broadcast,
      xnn_f32_igemm_ukernel_1x8__avx_broadcast,
      5 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX);
  }

  static void f32_gemm_6x8__avx_broadcast(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_6x8__avx_broadcast,
      xnn_f32_igemm_ukernel_6x8__avx_broadcast,
      xnn_f32_gemm_ukernel_1x8__avx_broadcast,
      xnn_f32_igemm_ukernel_1x8__avx_broadcast,
      6 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX);
  }

  static void f32_gemm_7x8__avx_broadcast(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_7x8__avx_broadcast,
      xnn_f32_igemm_ukernel_7x8__avx_broadcast,
      xnn_f32_gemm_ukernel_1x8__avx_broadcast,
      xnn_f32_igemm_ukernel_1x8__avx_broadcast,
      7 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX);
  }

  static void f32_gemm_4x8__fma3_broadcast(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_4x8__fma3_broadcast,
      xnn_f32_igemm_ukernel_4x8__fma3_broadcast,
      xnn_f32_gemm_ukernel_1x8__fma3_broadcast,
      xnn_f32_igemm_ukernel_1x8__fma3_broadcast,
      4 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckFMA3);
  }

  static void f32_gemm_5x8__fma3_broadcast(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_5x8__fma3_broadcast,
      xnn_f32_igemm_ukernel_5x8__fma3_broadcast,
      xnn_f32_gemm_ukernel_1x8__fma3_broadcast,
      xnn_f32_igemm_ukernel_1x8__fma3_broadcast,
      5 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckFMA3);
  }

  static void f32_gemm_6x8__fma3_broadcast(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_6x8__fma3_broadcast,
      xnn_f32_igemm_ukernel_6x8__fma3_broadcast,
      xnn_f32_gemm_ukernel_1x8__fma3_broadcast,
      xnn_f32_igemm_ukernel_1x8__fma3_broadcast,
      6 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckFMA3);
  }

  static void f32_gemm_7x8__fma3_broadcast(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_7x8__fma3_broadcast,
      xnn_f32_igemm_ukernel_7x8__fma3_broadcast,
      xnn_f32_gemm_ukernel_1x8__fma3_broadcast,
      xnn_f32_igemm_ukernel_1x8__fma3_broadcast,
      7 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckFMA3);
  }

  static void f32_gemm_8x8__fma3_broadcast(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_8x8__fma3_broadcast,
      xnn_f32_igemm_ukernel_8x8__fma3_broadcast,
      xnn_f32_gemm_ukernel_1x8__fma3_broadcast,
      xnn_f32_igemm_ukernel_1x8__fma3_broadcast,
      8 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckFMA3);
  }

  static void f32_gemm_4x16__avx512f_broadcast(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_4x16__avx512f_broadcast,
      xnn_f32_igemm_ukernel_4x16__avx512f_broadcast,
      xnn_f32_gemm_ukernel_1x16__avx512f_broadcast,
      xnn_f32_igemm_ukernel_1x16__avx512f_broadcast,
      4 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX512F);
  }

  static void f32_gemm_5x16__avx512f_broadcast(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_5x16__avx512f_broadcast,
      xnn_f32_igemm_ukernel_5x16__avx512f_broadcast,
      xnn_f32_gemm_ukernel_1x16__avx512f_broadcast,
      xnn_f32_igemm_ukernel_1x16__avx512f_broadcast,
      5 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX512F);
  }

  static void f32_gemm_6x16__avx512f_broadcast(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_6x16__avx512f_broadcast,
      xnn_f32_igemm_ukernel_6x16__avx512f_broadcast,
      xnn_f32_gemm_ukernel_1x16__avx512f_broadcast,
      xnn_f32_igemm_ukernel_1x16__avx512f_broadcast,
      6 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX512F);
  }

  static void f32_gemm_7x16__avx512f_broadcast(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_7x16__avx512f_broadcast,
      xnn_f32_igemm_ukernel_7x16__avx512f_broadcast,
      xnn_f32_gemm_ukernel_1x16__avx512f_broadcast,
      xnn_f32_igemm_ukernel_1x16__avx512f_broadcast,
      7 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX512F);
  }

  static void f32_gemm_8x16__avx512f_broadcast(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_8x16__avx512f_broadcast,
      xnn_f32_igemm_ukernel_8x16__avx512f_broadcast,
      xnn_f32_gemm_ukernel_1x16__avx512f_broadcast,
      xnn_f32_igemm_ukernel_1x16__avx512f_broadcast,
      8 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX512F);
  }

  BENCHMARK_CAPTURE(f32_gemm_4x8__sse_load1, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_4x8__sse_load1, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_4x8__sse_dup, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_4x8__sse_dup, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_4x8s4__sse, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_4x8s4__sse, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_4x8__avx_broadcast, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_4x8__avx_broadcast, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_5x8__avx_broadcast, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_5x8__avx_broadcast, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_6x8__avx_broadcast, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_6x8__avx_broadcast, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_7x8__avx_broadcast, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_7x8__avx_broadcast, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_4x8__fma3_broadcast, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_4x8__fma3_broadcast, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_5x8__fma3_broadcast, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_5x8__fma3_broadcast, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_6x8__fma3_broadcast, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_6x8__fma3_broadcast, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_7x8__fma3_broadcast, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_7x8__fma3_broadcast, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_8x8__fma3_broadcast, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_8x8__fma3_broadcast, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_4x16__avx512f_broadcast, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_4x16__avx512f_broadcast, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_5x16__avx512f_broadcast, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_5x16__avx512f_broadcast, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_6x16__avx512f_broadcast, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_6x16__avx512f_broadcast, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_7x16__avx512f_broadcast, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_7x16__avx512f_broadcast, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_8x16__avx512f_broadcast, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_8x16__avx512f_broadcast, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if !XNN_ARCH_WASM && !XNN_ARCH_ASMJS
  static void f32_gemm_4x8__psimd_loadsplat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_4x8__psimd_loadsplat,
      xnn_f32_igemm_ukernel_4x8__psimd_loadsplat,
      xnn_f32_gemm_ukernel_1x8__psimd_loadsplat,
      xnn_f32_igemm_ukernel_1x8__psimd_loadsplat,
      4 /* mr */, 8 /* nr */);
  }

  static void f32_gemm_6x8__psimd_loadsplat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_6x8__psimd_loadsplat,
      xnn_f32_igemm_ukernel_6x8__psimd_loadsplat,
      xnn_f32_gemm_ukernel_1x8__psimd_loadsplat,
      xnn_f32_igemm_ukernel_1x8__psimd_loadsplat,
      6 /* mr */, 8 /* nr */);
  }

  static void f32_gemm_4x8__psimd_splat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_4x8__psimd_splat,
      xnn_f32_igemm_ukernel_4x8__psimd_splat,
      xnn_f32_gemm_ukernel_1x8__psimd_splat,
      xnn_f32_igemm_ukernel_1x8__psimd_splat,
      4 /* mr */, 8 /* nr */);
  }

  static void f32_gemm_6x8__psimd_splat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_6x8__psimd_splat,
      xnn_f32_igemm_ukernel_6x8__psimd_splat,
      xnn_f32_gemm_ukernel_1x8__psimd_splat,
      xnn_f32_igemm_ukernel_1x8__psimd_splat,
      6 /* mr */, 8 /* nr */);
  }

  static void f32_gemm_4x8s4__psimd(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_4x8s4__psimd,
      xnn_f32_igemm_ukernel_4x8s4__psimd,
      xnn_f32_gemm_ukernel_1x8s4__psimd,
      xnn_f32_igemm_ukernel_1x8s4__psimd,
      4 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */);
  }

  static void f32_gemm_6x8s4__psimd(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_ukernel_6x8s4__psimd,
      xnn_f32_igemm_ukernel_6x8s4__psimd,
      xnn_f32_gemm_ukernel_1x8s4__psimd,
      xnn_f32_igemm_ukernel_1x8s4__psimd,
      6 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */);
  }

  BENCHMARK_CAPTURE(f32_gemm_4x8__psimd_loadsplat, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_4x8__psimd_loadsplat, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_6x8__psimd_loadsplat, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_6x8__psimd_loadsplat, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_4x8__psimd_splat, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_4x8__psimd_splat, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_6x8__psimd_splat, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_6x8__psimd_splat, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_4x8s4__psimd, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_4x8s4__psimd, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

  BENCHMARK_CAPTURE(f32_gemm_6x8s4__psimd, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
  BENCHMARK_CAPTURE(f32_gemm_6x8s4__psimd, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();
#endif  // !XNN_ARCH_WASM && !XNN_ARCH_ASMJS

static void f32_gemm_2x4__scalar(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_f32_gemm_ukernel_2x4__scalar,
    xnn_f32_igemm_ukernel_2x4__scalar,
    xnn_f32_gemm_ukernel_1x4__scalar,
    xnn_f32_igemm_ukernel_1x4__scalar,
    2 /* mr */, 4 /* nr */);
}

static void f32_gemm_4x4__scalar(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_f32_gemm_ukernel_4x4__scalar,
    xnn_f32_igemm_ukernel_4x4__scalar,
    xnn_f32_gemm_ukernel_1x4__scalar,
    xnn_f32_igemm_ukernel_1x4__scalar,
    4 /* mr */, 4 /* nr */);
}

BENCHMARK_CAPTURE(f32_gemm_2x4__scalar, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK_CAPTURE(f32_gemm_2x4__scalar, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

BENCHMARK_CAPTURE(f32_gemm_4x4__scalar, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK_CAPTURE(f32_gemm_4x4__scalar, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
