//
// Created by HP on 2026/6/11.
//

#ifndef GENERALDNNLIB_ZERO_TRT_CUDA_MEMORY_H
#define GENERALDNNLIB_ZERO_TRT_CUDA_MEMORY_H
#include "gdlz/export.h"
#include "gdlz/trt/core/cuda/trt_cuda_data.h"

namespace gdlz::trt::core::cuda {
    class GDLZ_CPP_API TensorRTCudaMemory{
        public:
            TensorRTCudaMemory()=default;
            ~TensorRTCudaMemory()=default;

            static int Malloc(int64_t size,TensorRTCudaAddress& data,DataType type);
            static int Malloc(int64_t size, void*& address, DataType type);
            static int Free(TensorRTCudaAddress& data);
            static int Free(const void*& data);
            static int Sync(TensorRTCoreCtx& core_ctx);

            template<typename T>
            static int Sync(T& ctx){
                return Sync(ctx.getCoreCtx());
            }
    };

}
#endif //GENERALDNNLIB_ZERO_TRT_CUDA_MEMORY_H