//
// Created by HP on 2026/11/2.
//

#ifndef GENERALDNNLIB_ZERO_TRT_CUDA_BUFFER_OP_H
#define GENERALDNNLIB_ZERO_TRT_CUDA_BUFFER_OP_H
#include <vector>
#include "trt_cuda_buffer.h"
#include "trt_cuda_buffer_type.h"
#include "gdlz/export.h"
#include "gdlz/trt/core/cuda/trt_cuda_data.h"

namespace gdlz::trt::core::cuda {

   GDLZ_CPP_API void*  resetSize   (const TensorRTCudaBuffer& buffer, const std::vector<int64_t>& shape,DataType dtype);
   GDLZ_CPP_API void*  resetSize   (const TensorRTCudaBuffer& buffer, const std::vector<int64_t>& shape,BufferType type);
   GDLZ_CPP_API void*  resetSize   (const TensorRTCudaBuffer& buffer, int64_t size);
   GDLZ_CPP_API void*  uploadData  (const TensorRTCudaBuffer& buffer, const std::vector<std::byte>& host);
   GDLZ_CPP_API void   downloadData(const TensorRTCudaBuffer& buffer);
}
#endif //GENERALDNNLIB_ZERO_TRT_CUDA_BUFFER_OP_H