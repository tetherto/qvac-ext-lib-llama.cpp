#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//------------------------------------------------------------------------------
// diag: src [N, 1, ne02, ne03] -> dst [N, N, ne02, ne03], off-diagonal = 0.
// Launch: global = (N, N, ne02 * ne03), local = small (e.g. 8x8x1).
//------------------------------------------------------------------------------
kernel void kernel_diag_f32(
        global const char * src0,
        ulong               offset0,
        global       char * dst,
        ulong               offsetd,
        int   N,
        int   ne02,
        ulong nb02_src, ulong nb03_src,
        ulong nb1_dst,  ulong nb2_dst,  ulong nb3_dst
) {
    src0 = src0 + offset0;
    dst  = dst  + offsetd;

    const int i0  = get_global_id(0);
    const int i1  = get_global_id(1);
    const int z   = get_global_id(2);

    if (i0 >= N || i1 >= N) return;

    const int i2 = z % ne02;
    const int i3 = z / ne02;

    global const float * s = (global const float *)(src0 + i3*nb03_src + i2*nb02_src);
    global       float * d = (global       float *)(dst  + i3*nb3_dst  + i2*nb2_dst + i1*nb1_dst);

    d[i0] = (i0 == i1) ? s[i1] : 0.0f;
}
