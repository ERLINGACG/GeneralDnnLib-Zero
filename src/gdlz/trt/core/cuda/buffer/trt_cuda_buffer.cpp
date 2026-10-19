//
// Created by HP on 2026/11/2.
//
#include "gdlz/trt/core/cuda/buffer/trt_cuda_buffer.h"
#include "gdlz/trt/core/cuda/buffer/private/trt_cuda_buffer_impl.h"
using gdlz::trt::core::cuda::TensorRTCudaBufferImpl;
using gdlz::trt::core::cuda::TensorRTCudaBuffer;

TensorRTCudaBuffer::TensorRTCudaBuffer() {
    buffer = std::make_unique<TensorRTCudaBufferImpl>();
}

TensorRTCudaBuffer::~TensorRTCudaBuffer() = default;
TensorRTCudaBuffer::TensorRTCudaBuffer(TensorRTCudaBuffer&&) noexcept = default;
TensorRTCudaBuffer& TensorRTCudaBuffer::operator=(TensorRTCudaBuffer&&) noexcept= default;

void* TensorRTCudaBuffer::host() const noexcept {
    return buffer->data();
}

size_t TensorRTCudaBuffer::size() const noexcept{
    return buffer->bytes();
}
