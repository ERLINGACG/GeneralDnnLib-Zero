//
// Created by HP on 2026/6/6.
//

#ifndef GENERALDNNLIB_ZERO_TRT_CUDA_POST_H
#define GENERALDNNLIB_ZERO_TRT_CUDA_POST_H


#include "../trt_cuda_data.h"
#include "gdlz/export.h"

namespace gdlz::trt::core::cuda {

        class GDLZ_CPP_API TensorRTCudaPost{
            public:
                TensorRTCudaPost() = default;
                ~TensorRTCudaPost() = default;

            //   static int PostCuda(void* host ,int64_t size,TensorRTCudaAddress& data,DataType type);

               static int PostCuda(void* host, int64_t size, void*& address, cudaStream_t stream, DataType type);



        };


}
#endif //GENERALDNNLIB_ZERO_TRT_CUDA_POST_H