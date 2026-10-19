//
// Created by HP on 2026/6/9.
//

#include "gdlz/trt/core/cuda/op/trt_cuda_get.h"

#include <cuda_runtime_api.h>
#include <iostream>

#include "gdlz/trt/core/trt_code.h"
using gdlz::trt::core::cuda::TensorRTCudaGet;



int TensorRTCudaGet::GetCudaData(TensorRTCudaData& data,
                                 const void* address,
                                 int64_t len,
                                 DataType type,
                                 cudaStream_t* stream)
{

    if (data.data==nullptr ){ return code::data::DATA_NULL;}

    if (address==nullptr){ return code::address::ADDRESS_NULL;}
    cudaError_t err = cudaMemcpyAsync(data.data, address,
                                    len * static_cast<size_t>(type),
                                    cudaMemcpyDeviceToHost, *stream
    );
    if (err != cudaSuccess){
        std::cerr<<"cudaMemcpyAsync failed:"<<err<<" size:"<<len<<std::endl;
        return code::cuda::AYS_FAILED;
    };
    err = cudaStreamSynchronize(*stream);
    if (err != cudaSuccess){
        std::cerr<<"cudaStreamSynchronize failed"<<std::endl;
        return code::cuda::SYN_FAILED;
    };
    data.len = len;
    return code::operation::SUCCESS;
}




int TensorRTCudaGet::GetCudaFLOAT32(TensorRTCudaData& data, const void* address, int64_t len, cudaStream_t* stream) {
    if (address==nullptr){ return code::address::ADDRESS_NULL;}
    if (data.data!=nullptr){

        delete[] static_cast<float*>(data.data);  // 先转换，再删除
        data.data=new float[len];

    }else{
        data.data=new float[len];
    }
    return GetCudaData(data,address,len,DataType::FLOAT32,stream);
}



