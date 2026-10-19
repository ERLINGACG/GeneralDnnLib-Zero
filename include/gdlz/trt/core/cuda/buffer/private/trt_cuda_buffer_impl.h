//
// Created by HP on 2026/11/2.
//

#ifndef GENERALDNNLIB_ZERO_TRT_CUDA_BUFFER_IMPL_H
#define GENERALDNNLIB_ZERO_TRT_CUDA_BUFFER_IMPL_H
#include <mutex>
#include <vector>

#include <cuda_runtime.h>



namespace gdlz::trt::core::cuda {

    struct TensorRTCudaBufferImpl{

        std::vector<std::byte> cpu_host;

        void*        gpu_address  = nullptr;
        size_t       gpu_capacity = 0;    // GPU 字节数
        cudaStream_t stream       = nullptr;

        [[nodiscard]] const void* data() const noexcept {
            return cpu_host.empty() ? nullptr : cpu_host.data();
        }

        void* data() noexcept {
            return cpu_host.empty() ? nullptr : cpu_host.data();
        }


        [[nodiscard]] size_t bytes() const noexcept{
            return cpu_host.size();
        }

        // void setHostData(const std::byte* src, int64_t bytes, const int64_t elem_len = 0) {
        //     if (src == nullptr || bytes <= 0) {
        //         cpu_host.clear();
        //         return;
        //     }
        //     cpu_host.assign(src, src + static_cast<size_t>(bytes));  // 不复用 clear+memcpy
        // }

        void setStream(const cudaStream_t* stream){
            this->stream = *stream;
        }


    };

}
#endif //GENERALDNNLIB_ZERO_TRT_CUDA_BUFFER_IMPL_H