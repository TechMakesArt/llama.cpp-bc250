#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require

#if USE_SUBGROUP_ADD || USE_SUBGROUP_ADD_NO_SHMEM
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#endif

#include "mul_mat_vec_glu_iface.glsl"

// Push constants kept identical to the baseline mul_mat_vec path. The fusion
// flags are unused for the GLU variant (bias fusion does not compose with
// SWIGLU here), but the layout matches so the dispatcher can reuse code paths.
layout (push_constant) uniform parameter
{
    uint ncols;
    uint stride_a;
    uint stride_b;
    uint stride_d;

    uint batch_stride_a;
    uint batch_stride_b;
    uint batch_stride_d;

    uint fusion_flags;

    uint base_work_group_y;
    uint ne02;
    uint ne12;
    uint broadcast2;
    uint broadcast3;
} p;

void get_offsets(out uint a_offset, out uint b_offset, out uint d_offset) {
    const uint batch_idx = gl_WorkGroupID.y + p.base_work_group_y;

    uint batch_idx_a = 0;
    if (batch_idx != 0) {
        const uint i13 = batch_idx / p.ne12;
        const uint i12 = batch_idx % p.ne12;

        const uint i03 = i13 / p.broadcast3;
        const uint i02 = i12 / p.broadcast2;

        batch_idx_a = i03 * p.ne02 + i02;
    }

    a_offset = batch_idx_a * (p.batch_stride_a / QUANT_K);
    b_offset = batch_idx * p.batch_stride_b;
    d_offset = batch_idx * p.batch_stride_d;
}

layout (constant_id = 0) const uint BLOCK_SIZE = 32;
layout (constant_id = 1) const uint NUM_ROWS = 1;
layout (constant_id = 2) const uint NUM_COLS = 1;

// SiLU(a) * b — the SWIGLU_SPLIT combination. Matches the `op()` in swiglu.comp.
FLOAT_TYPE swiglu_combine(FLOAT_TYPE a, FLOAT_TYPE b) {
    return a / (FLOAT_TYPE(1.0) + exp(-a)) * b;
}

#ifdef USE_SUBGROUP_ADD_NO_SHMEM
void reduce_result_glu(inout FLOAT_TYPE temp_gate[NUM_COLS][NUM_ROWS],
                       inout FLOAT_TYPE temp_up  [NUM_COLS][NUM_ROWS],
                       const in uint32_t d_offset, const in uint32_t first_row,
                       const in uint32_t num_rows, const in uint32_t tid) {
    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
        [[unroll]] for (uint n = 0; n < num_rows; ++n) {
            temp_gate[j][n] = subgroupAdd(temp_gate[j][n]);
            temp_up  [j][n] = subgroupAdd(temp_up  [j][n]);
        }
    }

    if (tid == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                const FLOAT_TYPE v = swiglu_combine(temp_gate[j][n], temp_up[j][n]);
                data_d[j*p.batch_stride_d + d_offset + first_row + n] = D_TYPE(v);
            }
        }
    }
}
#else
shared FLOAT_TYPE tmpsh_gate[NUM_COLS][NUM_ROWS][BLOCK_SIZE];
shared FLOAT_TYPE tmpsh_up  [NUM_COLS][NUM_ROWS][BLOCK_SIZE];

void reduce_result_glu(FLOAT_TYPE temp_gate[NUM_COLS][NUM_ROWS],
                       FLOAT_TYPE temp_up  [NUM_COLS][NUM_ROWS],
                       const in uint32_t d_offset, const in uint32_t first_row,
                       const in uint32_t num_rows, const in uint32_t tid) {
#if USE_SUBGROUP_ADD
    // Sum partials within each subgroup.
    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
        [[unroll]] for (uint n = 0; n < num_rows; ++n) {
            temp_gate[j][n] = subgroupAdd(temp_gate[j][n]);
            temp_up  [j][n] = subgroupAdd(temp_up  [j][n]);
        }
    }

    // Cross-subgroup via shared memory.
    if (gl_SubgroupInvocationID == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                tmpsh_gate[j][n][gl_SubgroupID] = temp_gate[j][n];
                tmpsh_up  [j][n][gl_SubgroupID] = temp_up  [j][n];
            }
        }
    }
    barrier();
    if (tid == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                FLOAT_TYPE g = FLOAT_TYPE(0);
                FLOAT_TYPE u = FLOAT_TYPE(0);
                [[unroll]] for (uint s = 0; s < gl_NumSubgroups; ++s) {
                    g += tmpsh_gate[j][n][s];
                    u += tmpsh_up  [j][n][s];
                }
                data_d[j*p.batch_stride_d + d_offset + first_row + n] = D_TYPE(swiglu_combine(g, u));
            }
        }
    }
#else
    // Pure-shmem reduction (no subgroup arithmetic).
    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
        [[unroll]] for (uint n = 0; n < num_rows; ++n) {
            tmpsh_gate[j][n][tid] = temp_gate[j][n];
            tmpsh_up  [j][n][tid] = temp_up  [j][n];
        }
    }
    barrier();
    [[unroll]] for (uint s = BLOCK_SIZE/2; s > 0; s >>= 1) {
        if (tid < s) {
            [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
                [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                    tmpsh_gate[j][n][tid] += tmpsh_gate[j][n][tid + s];
                    tmpsh_up  [j][n][tid] += tmpsh_up  [j][n][tid + s];
                }
            }
        }
        barrier();
    }
    if (tid == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                data_d[j*p.batch_stride_d + d_offset + first_row + n] =
                    D_TYPE(swiglu_combine(tmpsh_gate[j][n][0], tmpsh_up[j][n][0]));
            }
        }
    }
#endif
}
#endif
