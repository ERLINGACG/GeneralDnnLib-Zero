//
// Created by HP on 2026/6/12.
//
#include "gdlz/trt/illm/sample/cu/trt_illm_temp.cuh"

#include <iostream>


__global__ void ArgmaxKernelFP32(
    const float* logits,          // 指向整个 logits 张量的起始地址
    const int64_t start_token_pos,// 起始偏移（单位：元素个数）
    const int64_t local_size,     // 要处理的元素个数（即 vocab_size）
    int64_t* index                // 输出相对索引
) {
    auto tid = threadIdx.x;
    auto stride = blockDim.x;

    __shared__ float   s_vals[256];
    __shared__ int64_t s_idxs[256];

    // 定位到起始地址
    const float* local_logits = logits + start_token_pos;

    float max_val = -FLT_MAX;
    int64_t max_idx = -1;

    // 注意：循环范围是 0 ~ local_size-1，使用相对索引 i
    for (int64_t i = tid; i < local_size; i += stride) {
        float v = local_logits[i];   // 读取从 start_token_pos 开始的数据
        if (v > max_val) {
            max_val = v;
            max_idx = i;             // 保存相对偏移（0 ~ local_size-1）
        }
    }

    s_vals[tid] = max_val;
    s_idxs[tid] = max_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_vals[tid + s] > s_vals[tid]) {
                s_vals[tid] = s_vals[tid + s];
                s_idxs[tid] = s_idxs[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        index[0] = s_idxs[0];   // 输出相对索引
    }
}


int gdlz::trt::illm::sample::cu::Argmax(const void* logits,int64_t start_token_pos,int64_t loca_v)
{

    int64_t* d_index;
    cudaMalloc(&d_index, sizeof(int64_t));
    ArgmaxKernelFP32<<<1, 256>>>(
        static_cast<const float*>(logits),
        start_token_pos,
        loca_v,
        d_index
    );
    cudaDeviceSynchronize();  // 等待核函数完成

    int64_t max_idx = -1;
    cudaMemcpy(&max_idx, d_index, sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_index);
    return max_idx;
}
int64_t gdlz::trt::illm::sample::cu::Argmax(const void* logits, int64_t start_token_pos, int64_t loca_v, int64_t*& output_address) {

    cudaMalloc(&output_address, sizeof(int64_t));
    ArgmaxKernelFP32<<<1, 256>>>(
        static_cast<const float*>(logits),
        start_token_pos,
        loca_v,
        output_address
    );
    cudaDeviceSynchronize();  // 等待核函数完成

    int64_t max_idx = -1;
    cudaMemcpy(&max_idx, output_address, sizeof(int64_t), cudaMemcpyDeviceToHost);
    return max_idx;
}