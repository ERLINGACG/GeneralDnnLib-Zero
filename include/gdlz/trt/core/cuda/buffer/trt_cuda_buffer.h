//
// Created by HP on 2026/11/2.
//

#ifndef GENERALDNNLIB_ZERO_TRT_CUDA_BUFFER_H
#define GENERALDNNLIB_ZERO_TRT_CUDA_BUFFER_H
#include <memory>
#include <vector>
#include "gdlz/export.h"
namespace gdlz::trt::core::cuda {

    struct TensorRTCudaBufferImpl;

    struct GDLZ_CPP_API TensorRTCudaBuffer {

        std::unique_ptr<TensorRTCudaBufferImpl> buffer;

        TensorRTCudaBuffer();
        ~TensorRTCudaBuffer();

        TensorRTCudaBuffer(TensorRTCudaBuffer&&) noexcept;
        TensorRTCudaBuffer& operator=(TensorRTCudaBuffer&&) noexcept ;
        TensorRTCudaBuffer(const TensorRTCudaBuffer&)            = delete;
        TensorRTCudaBuffer& operator=(const TensorRTCudaBuffer&) = delete;

        [[nodiscard]] void*  host() const noexcept;
        [[nodiscard]] size_t size() const noexcept;

    };
}
#endif //GENERALDNNLIB_ZERO_TRT_CUDA_BUFFER_H