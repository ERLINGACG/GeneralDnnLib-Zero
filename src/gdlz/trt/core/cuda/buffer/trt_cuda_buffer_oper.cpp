//
// Created by HP on 2026/11/2.
//
#include "gdlz/trt/core/cuda/buffer/trt_cuda_buffer_oper.h"

#include <iostream>
#include <ostream>

#include "gdlz/trt/core/cuda/buffer/private/trt_cuda_buffer_impl.h"
using gdlz::trt::core::cuda::TensorRTCudaBufferImpl;

void* gdlz::trt::core::cuda::resetSize(const TensorRTCudaBuffer& buffer, const std::vector<int64_t>& shape, DataType dtype) {
    auto& impl  = *buffer.buffer;
    size_t need = 1;
    for (const auto s : shape) {
        if (s <= 0) {
            continue;
        }
        need *= s;
    }
    need *= sizeof(dtype);
    if (impl.gpu_address == nullptr || need > impl.gpu_capacity) {
        if (impl.gpu_address) {
            cudaFree(impl.gpu_address);       // 先释放旧的
            impl.gpu_address  = nullptr;
            impl.gpu_capacity = 0;
        }
        if (cudaMalloc(&impl.gpu_address, need) != cudaSuccess) {
            impl.gpu_address  = nullptr;
            impl.gpu_capacity = 0;
            std::cerr << "cudaMalloc failed, bytes:" << need << '\n';
            return nullptr;
        }
        impl.gpu_capacity = need;
    }
    return impl.gpu_address;
}

void* gdlz::trt::core::cuda::resetSize(const TensorRTCudaBuffer& buffer, const std::vector<int64_t>& shape,BufferType type)
{
    int64_t need = 1;
    for (const auto s : shape) {
        if (s <= 0) {
            continue;
        }
        need *= s;
    }
    const size_t elem_size = std::visit([]<typename T0>(T0) -> size_t {
        return sizeof(T0::Type);
    }, type);

    need *= static_cast<int64_t>(elem_size);

    return resetSize(buffer, need);
}

void* gdlz::trt::core::cuda::resetSize(const TensorRTCudaBuffer& buffer, const int64_t size)
{
    auto& impl  = *buffer.buffer;
    const size_t need = size;
    if (impl.gpu_address == nullptr || need > impl.gpu_capacity) {
        if (impl.gpu_address) {
            cudaFree(impl.gpu_address);       // 先释放旧的
            impl.gpu_address  = nullptr;
            impl.gpu_capacity = 0;
        }
        if (cudaMalloc(&impl.gpu_address, need) != cudaSuccess) {
            impl.gpu_address  = nullptr;
            impl.gpu_capacity = 0;
            std::cerr << "cudaMalloc failed, bytes:" << need << '\n';
            return nullptr;
        }
        impl.gpu_capacity = need;
    }
    return impl.gpu_address;
}

void* gdlz::trt::core::cuda::uploadData(const TensorRTCudaBuffer& buffer, const std::vector<std::byte>& host){

    auto& impl = *buffer.buffer;
    if (host.empty()) {
        std::cerr << "uploadData: host is empty\n";
        return nullptr;
    }


    // 2. 按“字节数”判断 GPU 容量，用独立字段，不足才重新分配
    const size_t need = host.size();
    if (impl.gpu_address == nullptr || need > impl.gpu_capacity) {
        if (impl.gpu_address) {
            cudaFree(impl.gpu_address);       // 先释放旧的
            impl.gpu_address  = nullptr;
            impl.gpu_capacity = 0;
        }
        if (cudaMalloc(&impl.gpu_address, need) != cudaSuccess) {
            impl.gpu_address  = nullptr;
            impl.gpu_capacity = 0;
            std::cerr << "cudaMalloc failed, bytes:" << need << '\n';
            return nullptr;
        }
        impl.gpu_capacity = need;
    }

    cudaError_t err = cudaMemcpyAsync(impl.gpu_address, host.data(), need,
                                     cudaMemcpyHostToDevice, impl.stream);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpyAsync failed:" << err << " size:" << need << '\n';
        cudaFree(impl.gpu_address);
        impl.gpu_address  = nullptr;
        impl.gpu_capacity = 0;
        return nullptr;
    }

    err = cudaStreamSynchronize(impl.stream);
    if (err != cudaSuccess) {
        std::cerr << "cudaStreamSynchronize failed:" << err << '\n';
        cudaFree(impl.gpu_address);
        impl.gpu_address  = nullptr;
        impl.gpu_capacity = 0;
        return nullptr;
    }
    return impl.gpu_address;
}

void gdlz::trt::core::cuda::downloadData(const TensorRTCudaBuffer& buffer) {
    auto& impl = *buffer.buffer;
    if (impl.gpu_address == nullptr || impl.gpu_capacity == 0) {
        std::cerr << "downloadData: gpu_address is nullptr or gpu_capacity is 0\n";
        return;
    }

    impl.cpu_host.resize(impl.gpu_capacity);
    // 2. D2H
    cudaError_t err = cudaMemcpyAsync(impl.cpu_host.data(), impl.gpu_address,
                                      impl.gpu_capacity,
                                      cudaMemcpyDeviceToHost, impl.stream);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpyAsync D2H failed:" << err
                  << " size:" << impl.gpu_capacity << '\n';
        impl.cpu_host.clear();
        return;
    }

    // 3. 同步
    err = cudaStreamSynchronize(impl.stream);
    if (err != cudaSuccess) {
        std::cerr << "cudaStreamSynchronize failed:" << err << '\n';
        impl.cpu_host.clear();
        return;
    }
}
