#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

//------------------------------------------------------------------------------
// cumsum (along innermost dim 0). Per-row inclusive prefix sum.
// Single workgroup per row; assumes ne00 <= subgroup_size.
//------------------------------------------------------------------------------
#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_32
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_cumsum_f32_sg(
        global const char * src0,
        ulong               offset0,
        global       char * dst,
        ulong               offsetd,
        int   ne00,
        ulong nb01, ulong nb02, ulong nb03,
        ulong nb1,  ulong nb2,  ulong nb3
) {
    src0 = src0 + offset0;
    dst  = dst  + offsetd;

    const int i01 = get_group_id(0);
    const int i02 = get_group_id(1);
    const int i03 = get_group_id(2);
    const int lid = get_local_id(0);

    global const float * sx = (global const float *)(src0 + i03*nb03 + i02*nb02 + i01*nb01);
    global       float * dx = (global       float *)(dst  + i03*nb3  + i02*nb2  + i01*nb1);

    const float v   = (lid < ne00) ? sx[lid] : 0.0f;
    const float acc = sub_group_scan_inclusive_add(v);

    if (lid < ne00) {
        dx[lid] = acc;
    }
}
