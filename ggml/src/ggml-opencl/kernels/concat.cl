kernel void kernel_concat_f32(
    global  const char * src0,
    ulong                offset0,
    global  const char * src1,
    ulong                offset1,
    global        char * dst,
    ulong                offsetd,
    int             ne00,
    int             ne01,
    int             ne02,
    int             ne03,
    ulong           nb00,
    ulong           nb01,
    ulong           nb02,
    ulong           nb03,
    ulong           nb10,
    ulong           nb11,
    ulong           nb12,
    ulong           nb13,
    int             ne0,
    ulong           nb0,
    ulong           nb1,
    ulong           nb2,
    ulong           nb3,
    int             dim
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    const int i3 = get_group_id(2);
    const int i2 = get_group_id(1);
    const int i1 = get_group_id(0);

    int o[4] = {0, 0, 0, 0};
    o[dim] = dim == 0 ? ne00 : (dim == 1 ? ne01 : (dim == 2 ? ne02 : ne03));

    global const float * x;

    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
            x = (global const float *)(src0 + (i3       )*nb03 + (i2       )*nb02 + (i1       )*nb01 + (i0       )*nb00);
        } else {
            x = (global const float *)(src1 + (i3 - o[3])*nb13 + (i2 - o[2])*nb12 + (i1 - o[1])*nb11 + (i0 - o[0])*nb10);
        }

        global float * y = (global float *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

        *y = *x;
    }
}

//------------------------------------------------------------------------------
// Flat 1D launch variant: one thread per output element. Suited for cases where
// the original kernel_concat_f32 would launch with a tiny local size (e.g. ne0=4
// for the gated-DeltaNet conv-input concat, where the existing dispatcher caps
// local_size = MIN(64, ne0) and ends up with 4 lanes/wave). For the flat path
// we launch a 2D global of (ne0*ne1, ne2*ne3) so each workgroup of 64 lanes
// always processes 64 contiguous output elements.
//------------------------------------------------------------------------------
kernel void kernel_concat_f32_flat(
    global  const char * src0,
    ulong                offset0,
    global  const char * src1,
    ulong                offset1,
    global        char * dst,
    ulong                offsetd,
    int             ne00,
    int             ne01,
    int             ne02,
    int             ne03,
    ulong           nb00,
    ulong           nb01,
    ulong           nb02,
    ulong           nb03,
    ulong           nb10,
    ulong           nb11,
    ulong           nb12,
    ulong           nb13,
    int             ne0,
    int             ne1,
    int             ne2,
    int             ne3,
    ulong           nb0,
    ulong           nb1,
    ulong           nb2,
    ulong           nb3,
    int             dim
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    const int gid01 = get_global_id(0);   // packs (i1 * ne0 + i0)
    const int gid23 = get_global_id(1);   // packs (i3 * ne2 + i2)

    if (gid01 >= ne0 * ne1 || gid23 >= ne2 * ne3) return;

    const int i0 = gid01 % ne0;
    const int i1 = gid01 / ne0;
    const int i2 = gid23 % ne2;
    const int i3 = gid23 / ne2;

    int o[4] = {0, 0, 0, 0};
    o[dim] = dim == 0 ? ne00 : (dim == 1 ? ne01 : (dim == 2 ? ne02 : ne03));

    global const float * x;
    if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
        x = (global const float *)(src0 + (i3       )*nb03 + (i2       )*nb02 + (i1       )*nb01 + (i0       )*nb00);
    } else {
        x = (global const float *)(src1 + (i3 - o[3])*nb13 + (i2 - o[2])*nb12 + (i1 - o[1])*nb11 + (i0 - o[0])*nb10);
    }
    global float * y = (global float *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);
    *y = *x;
}
