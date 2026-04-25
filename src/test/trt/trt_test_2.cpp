#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>
#include "NvInfer.h"
#include "cuda_runtime_api.h"

using namespace nvinfer1;

class SilentLogger : public ILogger {
    void log(Severity, const char*) noexcept override {}
} gLogger;

#define CHECK_CUDA(status) \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(status) << std::endl; \
        return -1; \
    }

int main() {
    // 你的 Token IDs
    std::vector<int> tokens = {
        31380, 55951, 15120, 87502, 50338, 69962, 25287, 5486,
        45736, 53434, 31968, 39607, 73981, 78244, 1811
    };

    const int MAX_SEQ = 512;
    // 3个输入
    std::vector<int32_t> input_ids(MAX_SEQ, 0);
    std::vector<int32_t> token_type_ids(MAX_SEQ, 0);
    std::vector<int32_t> attention_mask(MAX_SEQ, 1);

    memcpy(input_ids.data(), tokens.data(), tokens.size() * sizeof(int32_t));

    // 加载 TRT 引擎
    std::ifstream f("model.trt", std::ios::binary);
    std::string data((std::istreambuf_iterator<char>(f)), {});

    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(data.data(), data.size());
    IExecutionContext* ctx = engine->createExecutionContext();

    // ===================== GPU 内存分配（3输入 + 2输出）=====================
    void* d_input_ids = nullptr;
    void* d_token_type_ids = nullptr;
    void* d_attention_mask = nullptr;
    void* d_output = nullptr;
    void* d_677 = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input_ids, MAX_SEQ * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_token_type_ids, MAX_SEQ * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_attention_mask, MAX_SEQ * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_output, 512 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_677, 512 * sizeof(float)));

    // 拷贝数据到 GPU
    CHECK_CUDA(cudaMemcpy(d_input_ids, input_ids.data(), MAX_SEQ * sizeof(int32_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_token_type_ids, token_type_ids.data(), MAX_SEQ * sizeof(int32_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_attention_mask, attention_mask.data(), MAX_SEQ * sizeof(int32_t), cudaMemcpyHostToDevice));

    // ===================== 绑定顺序（必须和模型顺序完全一致！）=====================
    void* bindings[] = {
        d_input_ids,       // 输入 0: input_ids
        d_token_type_ids,  // 输入 1: token_type_ids
        d_attention_mask,  // 输入 2: attention_mask
        d_output,          // 输出 0: output
        d_677              // 输出 1: 677
    };

    // 推理
    bool ok = ctx->executeV2(bindings);
    if (!ok) {
        std::cerr << "executeV2 failed\n";
        return -1;
    }

    // 读取结果（我们只用第一个输出）
    std::vector<float> embedding(512);
    CHECK_CUDA(
        cudaMemcpy(embedding.data(), d_output, 512 * sizeof(float), cudaMemcpyDeviceToHost));

    // 输出前10个值
    std::cout << "✅ 推理成功！句向量前10个值：\n";
    for (int i = 0; i < 10; i++) {
        std::cout << embedding[i] << " ";
    }
    std::cout << "\n";

    // 释放资源
    cudaFree(d_input_ids);
    cudaFree(d_token_type_ids);
    cudaFree(d_attention_mask);
    cudaFree(d_output);
    cudaFree(d_677);
    delete ctx;
    delete engine;
    delete runtime;

    return 0;
}