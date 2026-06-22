//
// Created by HP on 2026/6/6.
//

#include "../../../../../../include/gdlz/trt/core/cuda/op/trt_cuda_post.h"

#include <cuda_runtime_api.h>
#include "gdlz/trt/core/trt_code.h"
#include <iostream>

using gdlz::trt::core::cuda::TensorRTCudaPost;

int TensorRTCudaPost::PostCuda(void* host, int64_t size, TensorRTCudaAddress& data, DataType type)
{
    if (data.address) {
        cudaFree(data.address);
        data.address = nullptr;
    }

    if (data.stream==nullptr){
        data.stream=new cudaStream_t();
        cudaStreamCreate(data.stream);
    }

    auto ret=cudaMalloc(&data.address,size * static_cast<size_t>(type));
    if (ret != cudaSuccess) {
        return code::cuda::MALLOC_FAILED;
    }

    ret=cudaMemcpyAsync(data.address, host, size * static_cast<size_t>(type),cudaMemcpyHostToDevice, *data.stream);
    if (ret != cudaSuccess) {
        cudaFree(data.address);
        data.address = nullptr;
        return code::cuda::AYS_FAILED;
    }
    data.org_array_len = size;
    return code::operation::SUCCESS;
}

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


int TensorRTCudaPost::PostCudaINT32(int32_t* host, int64_t size,TensorRTCudaAddress& data){
    return  PostCuda(host,size,data,DataType::INT32);
}


int TensorRTCudaPost::PostCudaINT64(int64_t* host, int64_t size, TensorRTCudaAddress& data){
    return  PostCuda(host,size,data,DataType::INT64);
}

int TensorRTCudaPost::PostCudaFLOAT32(float* host, int64_t size, TensorRTCudaAddress& data){
    return  PostCuda(host,size,data,DataType::FLOAT32);
}


int gdlz::trt::core::cuda::PostCudaINT32(int32_t* host, int64_t size, TensorRTCudaAddress& data){
    return TensorRTCudaPost::PostCudaINT32(host, size, data);
}

int  gdlz::trt::core::cuda::PostCudaINT64(int64_t* host, int64_t size, TensorRTCudaAddress& data){
    return TensorRTCudaPost::PostCudaINT64(host,size,data);
}

int gdlz::trt::core::cuda::PostCudaFLOAT32(float* host, int64_t size, TensorRTCudaAddress& data){
    return TensorRTCudaPost::PostCudaFLOAT32(host, size, data);
}

