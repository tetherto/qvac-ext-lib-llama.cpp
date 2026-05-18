#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

//------------------------------------------------------------------------------
// Gated DeltaNet autoregressive (n_tokens=1) recurrence, fused into one kernel.
//
// Inputs (per (h, n)):
//   s_in   [S, S, H, N]  prior state (untransposed)
//   q      [S, 1, H, N]  query
//   k      [S, 1, H, N]  key
//   v      [S, 1, H, N]  value
//   g      [g_ne0, 1, H, N]  raw gate (we apply exp() inside the kernel)
//          g_ne0 == 1 (GDA) or g_ne0 == S (KDA)
//   beta   [1, 1, H, N]   scalar gate per (h, n)
//
// Outputs (per (h, n)):
//   o      [S, 1, H, N]
//   s_out  [S, S, H, N]  next state
//
// Math (dropping h, n; all per-head):
//   s_g[j, i] = s_in[j, i] * exp(g[i if KDA else 0])   // dim 1 of s gets gated
//   sk[j]     = sum_i s_g[j, i] * k[i]                  // matrix-vector
//   sq[j]     = sum_i s_g[j, i] * q[i]                  // matrix-vector
//   d[j]      = beta * (v[j] - sk[j])                   // residual
//   kq        = sum_b k[b] * q[b]                       // dot product (no gate)
//   o[j]      = sq[j] + d[j] * kq
//   s_out[j, i] = s_g[j, i] + d[j] * k[i]               // rank-1 update
//
// Memory layout: s_in[j, i, h, n] is at index ((n*H + h)*S + i)*S + j
// (ggml: dim 0 fastest, so j varies fastest).
//
// Launch: 1 workgroup per (h, n), S threads per workgroup. Hardcoded S = 128
// for Qwen3.5 (head_v_dim). The kernel uses 2 Adreno-64 subgroups.
//------------------------------------------------------------------------------
#define DNET_AR_S 128

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#elif defined(INTEL_GPU)
REQD_SUBGROUP_SIZE_32
#endif
kernel void kernel_dnet_ar_f32_s128(
    global const char * s_in,   ulong off_s_in,
    global const char * q_g,    ulong off_q,
    global const char * k_g,    ulong off_k,
    global const char * v_g,    ulong off_v,
    global const char * g_g,    ulong off_g,
    global const char * b_g,    ulong off_b,
    global       char * o_g,    ulong off_o,
    global       char * s_out,  ulong off_s_out,
    int H,
    int g_ne0    // 1 (GDA) or S (KDA)
) {
    const int S = DNET_AR_S;
    const int h = get_group_id(1);
    const int n = get_group_id(2);
    const int tid = get_local_id(0);  // 0..S-1
    const int hn = n * H + h;

    global const float * s_in_p = (global const float *)(s_in + off_s_in) + (size_t)hn * S * S;
    global const float * k_p    = (global const float *)(k_g + off_k)     + (size_t)hn * S;
    global const float * q_p    = (global const float *)(q_g + off_q)     + (size_t)hn * S;
    global const float * v_p    = (global const float *)(v_g + off_v)     + (size_t)hn * S;
    global const float * b_p    = (global const float *)(b_g + off_b)     + (size_t)hn;
    global const float * g_p    = (global const float *)(g_g + off_g)     + (size_t)hn * g_ne0;
    global       float * o_p    = (global       float *)(o_g + off_o)     + (size_t)hn * S;
    global       float * s_out_p= (global       float *)(s_out + off_s_out) + (size_t)hn * S * S;

    __local float k_l[DNET_AR_S];
    __local float q_l[DNET_AR_S];
    __local float v_l[DNET_AR_S];
    __local float d_l[DNET_AR_S];
    __local float g_l[DNET_AR_S];   // valid for g_ne0 ∈ {1, S}; only g_l[0] used when g_ne0 == 1
    __local float beta_v;
    __local float partial_kq[2];     // 2 subgroups for S=128, sgs=64

    k_l[tid] = k_p[tid];
    q_l[tid] = q_p[tid];
    v_l[tid] = v_p[tid];
    if (g_ne0 == 1) {
        if (tid == 0) g_l[0] = g_p[0];
    } else {
        g_l[tid] = g_p[tid];
    }
    if (tid == 0) beta_v = b_p[0];

    barrier(CLK_LOCAL_MEM_FENCE);

    // sk[j=tid] = sum_i s[j, i] * exp(g(i)) * k[i]
    // sq[j=tid] = sum_i s[j, i] * exp(g(i)) * q[i]
    float sk_j = 0.0f;
    float sq_j = 0.0f;
    for (int i = 0; i < S; ++i) {
        const float g_v   = (g_ne0 == 1) ? g_l[0] : g_l[i];
        const float gate  = exp(g_v);
        const float s_ji  = s_in_p[tid + i * S];
        sk_j += s_ji * gate * k_l[i];
        sq_j += s_ji * gate * q_l[i];
    }

    const float dj = beta_v * (v_l[tid] - sk_j);
    d_l[tid] = dj;

    // kq = sum_b k[b] * q[b] (no gate)
    const float kq_partial = sub_group_reduce_add(k_l[tid] * q_l[tid]);
    if (get_sub_group_local_id() == 0) {
        partial_kq[get_sub_group_id()] = kq_partial;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const float kq = partial_kq[0] + partial_kq[1];

    // Output o[j=tid]
    o_p[tid] = sq_j + dj * kq;

    // Update state: s_out[j=tid, i] = s[j, i] * exp(g(i)) + d[j] * k[i]   for i = 0..S-1
    for (int i = 0; i < S; ++i) {
        const float g_v  = (g_ne0 == 1) ? g_l[0] : g_l[i];
        const float gate = exp(g_v);
        const float s_ji = s_in_p[tid + i * S];
        s_out_p[tid + i * S] = s_ji * gate + d_l[tid] * k_l[i];
    }
}
