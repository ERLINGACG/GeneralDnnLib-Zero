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

int TensorRTCudaGet::GetCudaData(TensorRTCudaData& data,
                                 const void* address,
                                 int64_t len,
                                 DataType type,
                                 cudaStream_t* stream)
{
    // if (data.data == nullptr) return code::data::DATA_NULL;
    // if (address == nullptr)   return code::address::ADDRESS_NULL;
    //
    // const size_t bytes = static_cast<size_t>(len) * static_cast<size_t>(type);
    //
    // // ---- 计时用 event ----
    // cudaEvent_t e0, e1, e2;
    // cudaEventCreate(&e0);
    // cudaEventCreate(&e1);
    // cudaEventCreate(&e2);
    //
    // // e0: 拷贝开始前（此时 stream 上可能还排着推理 kernel）
    // cudaEventRecord(e0, *stream);
    //
    // cudaError_t err = cudaMemcpyAsync(data.data, address, bytes,
    //                                   cudaMemcpyDeviceToHost, *stream);
    // if (err != cudaSuccess) {
    //     std::cerr << "cudaMemcpyAsync failed: " << err
    //               << " size: " << bytes << std::endl;
    //     return code::cuda::AYS_FAILED;
    // }
    //
    // // e1: 拷贝任务提交到 stream 之后（GPU 上拷贝完成的位置）
    // cudaEventRecord(e1, *stream);
    //
    // err = cudaStreamSynchronize(*stream);
    // if (err != cudaSuccess) {
    //     std::cerr << "cudaStreamSynchronize failed" << std::endl;
    //     return code::cuda::SYN_FAILED;
    // }
    //
    // // e2: sync 返回后
    // cudaEventRecord(e2, *stream);
    // cudaEventSynchronize(e2);
    //
    // float wait_ms = 0, copy_ms = 0;
    // cudaEventElapsedTime(&wait_ms, e0, e1);  // 从“拷贝开始”到“拷贝完成”的 GPU 时间
    // cudaEventElapsedTime(&copy_ms, e1, e2);  // sync 本身的空转，正常接近 0
    //
    // std::cout << "GPU wait+copy: " << wait_ms << " ms, "
    //           << "sync overhead: " << copy_ms << " ms, "
    //           << "bytes: " << bytes << std::endl;
    //
    // cudaEventDestroy(e0);
    // cudaEventDestroy(e1);
    // cudaEventDestroy(e2);
    //
    // data.len = len;
    // return code::operation::SUCCESS;
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

int TensorRTCudaGet::GetCudaFLOAT32(TensorRTCudaData& data, const void* address, int64_t len, cudaStream_t* stream) {
    if (address==nullptr){ return code::address::ADDRESS_NULL;}
    if (data.data!=nullptr){

        delete[] static_cast<float*>(data.data);  // 先转换，再删除
        data.data=new float[len];

    }else{  data.data=new float[len]; }
    return GetCudaData(data,address,len,DataType::FLOAT32,stream);
}


int  gdlz::trt::core::cuda::GetCudaINT32 (TensorRTCudaData& data, TensorRTCudaAddress& address){
    return TensorRTCudaGet::GetCudaINT32(data, address);
}
int  gdlz::trt::core::cuda::GetCudaINT64 (TensorRTCudaData& data, TensorRTCudaAddress& address){
    return TensorRTCudaGet::GetCudaINT64(data,address);
}
