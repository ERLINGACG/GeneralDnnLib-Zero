//
// Created by HP on 2026/6/9.
//

#include "gdlz/trt/core/cuda/op/trt_cuda_address.h"

int gdlz::trt::core::cuda::SetAddressStream(TensorRTCudaAddress& dit, const TensorRTCudaAddress& src){
    dit.stream=src.stream;
    return 0;
}

