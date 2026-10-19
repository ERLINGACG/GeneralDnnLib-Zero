//
// Created by HP on 2026/6/6.
//

#include "../../../../../../include/gdlz/trt/core/cuda/op/trt_cuda_post.h"

#include <cuda_runtime_api.h>
#include "gdlz/trt/core/trt_code.h"
#include <iostream>

using gdlz::trt::core::cuda::TensorRTCudaPost;



int TensorRTCudaPost::PostCuda(void* host, int64_t size, void*& address, cudaStream_t stream, DataType type)
{
    if (stream==nullptr){
        cudaStreamCreate(&stream);
    }
    if (address!=nullptr){
        cudaFree(address);
        address = nullptr;
    }
    auto ret=cudaMalloc(&address,size * static_cast<size_t>(type));
    if (ret != cudaSuccess) {
        return code::cuda::MALLOC_FAILED;
    }

    if (address==nullptr){ return code::address::ADDRESS_NULL;}
    ret=cudaMemcpyAsync(
        address,
        host,
        size * static_cast<size_t>(type),
        cudaMemcpyHostToDevice,
        stream
    );
    if (ret != cudaSuccess){
        return code::cuda::AYS_FAILED;
    }
    return code::operation::SUCCESS;
}



