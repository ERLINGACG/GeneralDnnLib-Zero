//
// Created by HP on 2026/6/9.
//

#ifndef GENERALDNNLIB_ZERO_TRT_CUDA_GET_H
#define GENERALDNNLIB_ZERO_TRT_CUDA_GET_H
#include "gdlz/export.h"
#include "gdlz/trt/core/trt_core_data.h"
#include "gdlz/trt/core/cuda/trt_cuda_data.h"

namespace gdlz::trt::core::cuda
{
    class GDLZ_CPP_API TensorRTCudaGet{

    public:
        static int GetCudaData (TensorRTCudaData& data,TensorRTCudaAddress& address,DataType type);
        static int GetCudaData (TensorRTCudaData& data, const void* address, int64_t len, DataType type, cudaStream_t* stream);
        static int GetCudaINT32(TensorRTCudaData& data,TensorRTCudaAddress& address);
        static int GetCudaINT64(TensorRTCudaData& data,TensorRTCudaAddress& address);

    };

    GDLZ_CORE_API int GetCudaINT32 (TensorRTCudaData& data,TensorRTCudaAddress& address);
    GDLZ_CORE_API int GetCudaINT64 (TensorRTCudaData& data,TensorRTCudaAddress& address);
}
#endif //GENERALDNNLIB_ZERO_TRT_CUDA_GET_H