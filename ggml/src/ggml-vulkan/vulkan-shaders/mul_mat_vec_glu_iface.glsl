#include "types.glsl"

// Binding layout for fused MUL_MAT(gate) + MUL_MAT(up) + SWIGLU_SPLIT.
//   0: gate weights (A1)
//   1: activation   (B, shared between both matmuls)
//   2: output       (D, single combined tensor, sized like one of the matmul outputs)
//   3: up weights   (A2)
//
// The gate and up weight matrices share identical quant format, shape, and scales
// structure — only the numeric data differs. So we declare the same packed aliases
// for both bindings.

layout (binding = 0) readonly buffer A1 {A_TYPE data_a1[];};
#if defined(A_TYPEV4)
layout (binding = 0) readonly buffer A1V4 {A_TYPEV4 data_a1_v4[];};
#endif
#if defined(A_TYPE_PACKED16)
layout (binding = 0) readonly buffer A1_PACKED16 {A_TYPE_PACKED16 data_a1_packed16[];};
#endif
#if defined(A_TYPE_PACKED32)
layout (binding = 0) readonly buffer A1_PACKED32 {A_TYPE_PACKED32 data_a1_packed32[];};
#endif

layout (binding = 1) readonly buffer B {B_TYPE data_b[];};
#ifdef B_TYPEV2
layout (binding = 1) readonly buffer BV2 {B_TYPEV2 data_b_v2[];};
#endif
#ifdef B_TYPEV4
layout (binding = 1) readonly buffer BV4 {B_TYPEV4 data_b_v4[];};
#endif

layout (binding = 2) writeonly buffer D {D_TYPE data_d[];};

layout (binding = 3) readonly buffer A2 {A_TYPE data_a2[];};
#if defined(A_TYPEV4)
layout (binding = 3) readonly buffer A2V4 {A_TYPEV4 data_a2_v4[];};
#endif
#if defined(A_TYPE_PACKED16)
layout (binding = 3) readonly buffer A2_PACKED16 {A_TYPE_PACKED16 data_a2_packed16[];};
#endif
#if defined(A_TYPE_PACKED32)
layout (binding = 3) readonly buffer A2_PACKED32 {A_TYPE_PACKED32 data_a2_packed32[];};
#endif
