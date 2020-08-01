#!/usr/bin/env python
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


def _indent(text):
  return "\n".join(map(lambda t: "  " + t if t else t, text.splitlines()))


def _remove_duplicate_newlines(text):
  filtered_lines = list()
  last_newline = False
  for line in text.splitlines():
    is_newline = len(line.strip()) == 0
    if not is_newline or not last_newline:
      filtered_lines.append(line)
    last_newline = is_newline
  return "\n".join(filtered_lines)


_ARCH_TO_MACRO_MAP = {
  "aarch32": "XNN_ARCH_ARM",
  "aarch64": "XNN_ARCH_ARM64",
  "x86-32": "XNN_ARCH_X86",
  "x86-64": "XNN_ARCH_X86_64",
  "wasm": "XNN_ARCH_WASM",
  "wasmsimd": "XNN_ARCH_WASMSIMD",
}

_ISA_TO_ARCH_MAP = {
  "neon": ["aarch32", "aarch64"],
  "neonfma": ["aarch32", "aarch64"],
  "neonv8": ["aarch32", "aarch64"],
  "neonfp16arith": ["aarch32", "aarch64"],
  "sse": ["x86-32", "x86-64"],
  "sse2": ["x86-32", "x86-64"],
  "ssse3": ["x86-32", "x86-64"],
  "sse41": ["x86-32", "x86-64"],
  "avx": ["x86-32", "x86-64"],
  "xop": ["x86-32", "x86-64"],
  "fma3": ["x86-32", "x86-64"],
  "avx2": ["x86-32", "x86-64"],
  "avx512f": ["x86-32", "x86-64"],
  "wasm": ["wasm", "wasmsimd"],
  "wasmsimd": ["wasmsimd"],
  "psimd": [],
}

_ISA_TO_CHECK_MAP = {
  "neon": "TEST_REQUIRES_ARM_NEON",
  "neonfma": "TEST_REQUIRES_ARM_NEON_FMA",
  "neonv8": "TEST_REQUIRES_ARM_NEON_V8",
  "neonfp16arith": "TEST_REQUIRES_ARM_NEON_FP16_ARITH",
  "sse": "TEST_REQUIRES_X86_SSE",
  "sse2": "TEST_REQUIRES_X86_SSE2",
  "ssse3": "TEST_REQUIRES_X86_SSSE3",
  "sse41": "TEST_REQUIRES_X86_SSE41",
  "avx": "TEST_REQUIRES_X86_AVX",
  "xop": "TEST_REQUIRES_X86_XOP",
  "avx2": "TEST_REQUIRES_X86_AVX2",
  "fma3": "TEST_REQUIRES_X86_FMA3",
  "avx512f": "TEST_REQUIRES_X86_AVX512F",
  "psimd": "TEST_REQUIRES_PSIMD",
}


def parse_target_name(target_name):
  arch = list()
  isa = None
  for target_part in target_name.split("_"):
    if target_part in _ARCH_TO_MACRO_MAP:
      if target_part in _ISA_TO_ARCH_MAP:
        arch = _ISA_TO_ARCH_MAP[target_part]
        isa = target_part
      else:
        arch = [target_part]
    elif target_part in _ISA_TO_ARCH_MAP:
      isa = target_part
  if isa and not arch:
    arch = _ISA_TO_ARCH_MAP[isa]

  return arch, isa


def generate_isa_check_macro(isa):
  return _ISA_TO_CHECK_MAP.get(isa, "")


def postprocess_test_case(test_case, arch, isa, assembly=False):
  test_case = _remove_duplicate_newlines(test_case)
  if arch:
    guard = " || ".join(map(_ARCH_TO_MACRO_MAP.get, arch))
    if assembly:
      guard += " && XNN_ENABLE_ASSEMBLY"
    return "#if %s\n" % guard + \
      _indent(test_case) + "\n" + \
      "#endif  // %s\n" % guard
  elif isa == "psimd":
    guard = "!XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC"
    return "#if %s\n" % guard + \
      _indent(test_case) + "\n" + \
      "#endif  // %s\n" % guard
  else:
    return test_case
