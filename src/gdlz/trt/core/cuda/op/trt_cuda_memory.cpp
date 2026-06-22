//
// Created by HP on 2026/6/11.
//

#include "gdlz/trt/core/cuda/op/trt_cuda_memory.h"

#include <cuda_runtime_api.h>

#include "gdlz/trt/core/trt_code.h"

int gdlz::trt::core::cuda::TensorRTCudaMemory::Malloc(int64_t size, TensorRTCudaAddress& data, DataType type)
{
    if (data.address!=nullptr){
        cudaFree(&data.address);
    }
    cudaMalloc(&data.address,size* static_cast<size_t>(type));
    data.org_array_len = size;
    return code::operation::SUCCESS;
}

int gdlz::trt::core::cuda::TensorRTCudaMemory::Malloc(int64_t size, void*& address, DataType type)
{
    cudaMalloc(&address,size* static_cast<size_t>(type));
    return code::operation::SUCCESS;
}

int gdlz::trt::core::cuda::TensorRTCudaMemory::Free(TensorRTCudaAddress& data)
{
    if (data.address!=nullptr){
        cudaFree(&data.address);
    }
    if (data.stream!=nullptr){
        cudaStreamDestroy(*data.stream);
        data.stream = nullptr;
    }
    return code::operation::SUCCESS;
}

int gdlz::trt::core::cuda::TensorRTCudaMemory::Free(const void*& data) {

    if (data!=nullptr){
        cudaFree(const_cast<void*>(data));
        data=nullptr;
    }
    return code::operation::SUCCESS;
}

int gdlz::trt::core::cuda::TensorRTCudaMemory::Sync(TensorRTCoreCtx& core_ctx)
{
    auto res=cudaStreamSynchronize(*core_ctx.stream);
    if (res!=cudaSuccess) {
        return code::cuda::SYN_FAILED;
    }
    return code::operation::SUCCESS;
}
