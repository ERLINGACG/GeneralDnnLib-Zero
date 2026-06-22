//
// Created by HP on 2026/6/9.
//

#ifndef GENERALDNNLIB_ZERO_TRT_CUDA_ADDRESS_H
#define GENERALDNNLIB_ZERO_TRT_CUDA_ADDRESS_H
#include "gdlz/export.h"
#include "gdlz/trt/core/trt_core_data.h"
#include "gdlz/trt/core/cuda/trt_cuda_data.h"

namespace gdlz::trt::core::cuda{

    GDLZ_CORE_API int SetAddressStream(TensorRTCudaAddress& dit, const TensorRTCudaAddress& src);

}
#endif //GENERALDNNLIB_ZERO_TRT_CUDA_ADDRESS_H