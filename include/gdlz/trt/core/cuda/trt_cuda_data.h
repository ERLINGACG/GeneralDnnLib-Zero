//
// Created by HP on 2026/6/6.
//

#ifndef GENERALDNNLIB_ZERO_TRT_CUDA_DATA_H
#define GENERALDNNLIB_ZERO_TRT_CUDA_DATA_H
#include <cstdint>
#include <driver_types.h>
#include <memory>

#include "gdlz/trt/core/trt_core_data.h"
#define CHECK_CUDA_STATUS(status) \
    if (status != cudaSuccess) { \
    std::cerr << "CUDA Error: " << cudaGetErrorString(status) << std::endl; \
    return -1; \
    }

namespace gdlz::trt::core::cuda{

    struct TensorRTCudaAddress{
        void* address;
        cudaStream_t* stream;
        int64_t org_array_len;
    };

    struct TensorRTCudaData{
        void*   data;
        int64_t len;
    };

    enum class DataType:int32_t{
        INT64   = static_cast<int32_t>(sizeof(int64_t)), //
        INT32   = static_cast<int32_t>(sizeof(int32_t)),
        INT16   = static_cast<int32_t>(sizeof(int16_t)),
        INT8    = static_cast<int32_t>(sizeof(int8_t)),
        INT4    = 1,   // 实际占 1 字节，但只使用低 4 位
        FLOAT32 = static_cast<int32_t>(sizeof(float)),
        FLOAT16 = static_cast<int32_t>(sizeof(uint16_t)), // 或直接 2
    };
}

#endif //GENERALDNNLIB_ZERO_TRT_CUDA_DATA_H