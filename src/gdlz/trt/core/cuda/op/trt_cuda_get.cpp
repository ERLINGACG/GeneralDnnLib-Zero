//
// Created by HP on 2026/6/9.
//

#include "gdlz/trt/core/cuda/op/trt_cuda_get.h"

#include <cuda_runtime_api.h>
#include <iostream>

#include "gdlz/trt/core/trt_code.h"
using gdlz::trt::core::cuda::TensorRTCudaGet;

int TensorRTCudaGet::GetCudaData(TensorRTCudaData& data, TensorRTCudaAddress& address,DataType type)
{
    if (data.data==nullptr){ return code::data::DATA_NULL;}

    cudaError_t err = cudaMemcpyAsync(data.data, address.address,
                                    address.org_array_len * static_cast<size_t>(type),
                                    cudaMemcpyDeviceToHost, *address.stream
    );
    if (err != cudaSuccess){
        std::cerr<<"cudaMemcpyAsync failed:"<<err<<" size:"<<address.org_array_len<<" type:int32_t"<<std::endl;
        return code::cuda::AYS_FAILED;
    };
    err = cudaStreamSynchronize(*address.stream);
    if (err != cudaSuccess){
        std::cerr<<"cudaStreamSynchronize failed"<<std::endl;
        return code::cuda::SYN_FAILED;
    };
    data.len = address.org_array_len;
    return code::operation::SUCCESS;
}

int TensorRTCudaGet::GetCudaData(TensorRTCudaData& data, const void* address, int64_t len, DataType type, cudaStream_t* stream)
{
    if (data.data==nullptr){ return code::data::DATA_NULL;}

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


int TensorRTCudaGet::GetCudaINT32(TensorRTCudaData& data, TensorRTCudaAddress& address) {
    if (!address.address){
        std::cerr<<"address.address is null"<<std::endl;
        return -1;
    }
    if (data.data==nullptr){
        data.data=new int32_t[address.org_array_len];
    }else {
        delete[] static_cast<int32_t*>(data.data);  // 先转换，再删除
        data.data=new int32_t[address.org_array_len];
    }
    return GetCudaData(data,address,DataType::INT32);
}

int TensorRTCudaGet::GetCudaINT64(TensorRTCudaData& data, TensorRTCudaAddress& address){
    if (!address.address) return code::address::ADDRESS_NULL;
    if (data.data!=nullptr){

        delete[] static_cast<int64_t*>(data.data);  // 先转换，再删除
        data.data=new int64_t[address.org_array_len];

    }else{  data.data=new int64_t[address.org_array_len]; }

    return GetCudaData(data,address,DataType::INT64);
}

int  gdlz::trt::core::cuda::GetCudaINT32 (TensorRTCudaData& data, TensorRTCudaAddress& address){
    return TensorRTCudaGet::GetCudaINT32(data, address);
}
int  gdlz::trt::core::cuda::GetCudaINT64 (TensorRTCudaData& data, TensorRTCudaAddress& address){
    return TensorRTCudaGet::GetCudaINT64(data,address);
}
