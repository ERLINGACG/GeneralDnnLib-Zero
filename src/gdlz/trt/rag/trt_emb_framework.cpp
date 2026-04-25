//
// Created by HP on 2026/4/19.
//
#include "gdlz/trt/rag/trt_emb_framework.h"

#include <iostream>
#include <ostream>
#include <dylog/logger.h>
#include <cuda_fp16.h>   // 提供 half / __half 类型
#include <ranges>
#include <unordered_map>
// 生成 TRT 引擎
// trtexec --onnx=model_1.onnx --minShapes=input_ids:1x1,token_type_ids:1x1,attention_mask:1x1 --optShapes=input_ids:1x128,token_type_ids:1x128,attention_mask:1x128 --maxShapes=input_ids:1x512,token_type_ids:1x512,attention_mask:1x512 --saveEngine=model_1.trt
void gdlz::trt::rag::TrtEmbFramework::ResourceDirector(TrtEmbDirector& director, const char* config_path)
{
    director.Handle(engine, config_path);
}

void gdlz::trt::rag::TrtEmbFramework::GetEngineInfo() const
{
    for (auto& info : engine.engine_info){
        dylog::DynamicLogger().info("tensorName: {}, shape: {}", info.first.c_str(), info.second.c_str());
    }
}

int gdlz::trt::rag::TrtEmbFramework::CreateCtx(TrtEmbCtx& ctx) const
{
    ctx.context.reset(engine.engine->createExecutionContext());
    if (ctx.context==nullptr)
    {
        dylog::DynamicLogger().error("create ExecutionContext failed");
        return -1;
    }
    return 0;
}

int gdlz::trt::rag::TrtEmbFramework::PostCuda(
    const TrtEmbCtx& ctx,
    const char* name,
    const void* input,
    int len,int isMask
) const {

    using namespace  nvinfer1;

    size_t put_size= 0;

    auto it = typeSizeMap.find(this->engine.engine->getTensorDataType(name));
    if (it == typeSizeMap.end()) {
        dylog::DynamicLogger().error("tensorName: {} unknown type", name);
        return -1;
    }
    auto binding = ctx.bindings->find(name);
    if (binding==ctx.bindings->end()){
        dylog::DynamicLogger().error("tensorName: {} not found", name);
        return -1;
    }
    put_size = it->second;
    if (ctx.context!=nullptr && ctx.bindings!=nullptr) {
        if (isMask==2){

            const std::vector mask(len, 1);
            CHECK_CUDA(
                cudaMalloc(&binding->second, len * put_size)
                );
            CHECK_CUDA(
                cudaMemcpy(binding->second, mask.data(), len * put_size,
                    cudaMemcpyHostToDevice
                )
            );
            dylog::DynamicLogger().info("tensorName: {}, shape: {} ,all 1", name,len);

        }else if (isMask==1){

            const std::vector mask(len, 0);
            CHECK_CUDA(
                cudaMalloc(&binding->second, len * put_size)
                );
            CHECK_CUDA(
                cudaMemcpy(binding->second, mask.data(), len * put_size,
                    cudaMemcpyHostToDevice
                )
            );
            dylog::DynamicLogger().info("tensorName: {}, shape: {} ,all 0", name,len);
        }else{
            CHECK_CUDA(
                cudaMalloc(&binding->second, len * put_size)
                );
            CHECK_CUDA(
                cudaMemcpy(binding->second, input, len * put_size,
                    cudaMemcpyHostToDevice
                )
            );
        }

    }
    return 0;
}

int gdlz::trt::rag::TrtEmbFramework::SetDims(TrtEmbCtx& ctx, const char* name, const int* input, int len)
{
    using namespace nvinfer1;
    Dims dims;
    dims.nbDims = len;
    for (int i = 0; i < len; ++i) dims.d[i] = input[i];

    bool ok = ctx.context->setInputShape(name, dims);
    if (ok) {
        if (ctx.bindings==nullptr) {
           ctx.bindings=std::make_unique<std::map<std::string, void*>>();
        }
        ctx.bindings->insert({name, nullptr});
        dylog::DynamicLogger().info("set Input shape success: {}",ok);
        return 0;
    }
    dylog::DynamicLogger().error("set Input shape failed: {}",ok);
    return -1;
}

int gdlz::trt::rag::TrtEmbFramework::GetDims(TrtEmbCtx& ctx, const char* name) const
{
    using namespace nvinfer1;
    Dims dims = ctx.context->getTensorShape(name);
    if (dims.nbDims==0) {
        dylog::DynamicLogger().error("tensorName: {} not found", name);
        return -1;
    }
    int64_t size =1;
    for (int i = 0; i < dims.nbDims; ++i) {
        size *= dims.d[i];
    }
    ctx.bindings->insert({name, nullptr});

    dylog::DynamicLogger().info("tensorName: {}, size: {}", name, size);

    auto binding = ctx.bindings->find(name);
    if (binding==ctx.bindings->end()){
        dylog::DynamicLogger().error("tensorName: {} not found", name);
        return -1;
    }
    auto type_size = typeSizeMap.find(this->engine.engine->getTensorDataType(name));
    if (type_size == typeSizeMap.end()) {
        dylog::DynamicLogger().error("tensorName: {} unknown type", name);
        return -1;
    }
    CHECK_CUDA(cudaMalloc(&binding->second, size * type_size->second));

    return 0;

}

int gdlz::trt::rag::TrtEmbFramework::Forward(TrtEmbCtx& ctx) const
{
    std::vector<void*> shapes;
    for (const auto& key : engine.engine_info | std::views::keys){
        shapes.push_back(ctx.bindings->at(key));
    }
    try{
        ctx.context->executeV2(shapes.data());
        return 0;
    }catch (const std::exception& e){
        dylog::DynamicLogger().error("executeV2 failed: {}", e.what());
        return -1;
    }
}

int gdlz::trt::rag::TrtEmbFramework::GetEmbedding(
    const TrtEmbCtx& ctx, TrtEmbData& embedding,const char* name
) const {

    auto type = this->engine.engine->getTensorDataType(name);
    auto dims = ctx.context->getTensorShape(name);
    auto type_size = typeSizeMap.find(type);
    if (type_size->second==0) {
        dylog::DynamicLogger().error("tensorName: {} unknown type", name);
        return -1;
    }

    int64_t len=1;
    for (int i = 0; i < dims.nbDims; ++i) {
        len *= dims.d[i];
    }

    std::vector<float> vec(len);
    auto binding = ctx.bindings->find(name);
    if (binding==ctx.bindings->end()){
        dylog::DynamicLogger().error("tensorName: {} not found", name);
        return -1;
    }
    CHECK_CUDA(cudaMemcpy(vec.data(), binding->second, vec.size() * type_size->second, cudaMemcpyDeviceToHost));
    embedding.embedding = std::make_unique<float[]>(vec.size());
    embedding.embedding_size = static_cast<int>(vec.size());
    std::ranges::copy(vec, embedding.embedding.get());
    return 0;
}


// std::vector<float> gdlz::trt::rag::TrtEmbFramework::Forward(const int* input, const int len) const
// {
//     using namespace nvinfer1;
//     auto cudaDeleter = [](void* ptr) { if (ptr) cudaFree(ptr); };
//     using CudaUniquePtr = std::unique_ptr<void, decltype(cudaDeleter)>;
//
//
//     auto ctx=engine.engine->createExecutionContext();
//     ctx->setInputShape("input_ids",      Dims2{1, len});
//     ctx->setInputShape("token_type_ids", Dims2{1, len});
//     ctx->setInputShape("attention_mask", Dims2{1, len});
//
//     dylog::DynamicLogger().info("set Input shape success");
//
//     void* d_input_ids_wrapper = nullptr;
//     void* d_token_type_ids_wrapper = nullptr;
//     void* d_attention_mask_wrapper = nullptr;
//
//     cudaMalloc(&d_input_ids_wrapper, len * sizeof(int));
//     cudaMalloc(&d_token_type_ids_wrapper, len * sizeof(int));
//     cudaMalloc(&d_attention_mask_wrapper, len * sizeof(int));
//
//     auto d_input_ids      = CudaUniquePtr(d_input_ids_wrapper, cudaDeleter);
//     auto d_token_type_ids = CudaUniquePtr(d_token_type_ids_wrapper, cudaDeleter);
//     auto d_attention_mask = CudaUniquePtr(d_attention_mask_wrapper, cudaDeleter);
//
//     cudaMemcpy(d_input_ids.get(), input, len * sizeof(int), cudaMemcpyHostToDevice);
//     // token_type_ids 通常全 0，attention_mask 全 1（长度 len）
//     std::vector token_type_ids(len, 0);
//     std::vector attention_mask(len, 1);
//
//     cudaMemcpy(d_token_type_ids.get(), token_type_ids.data(), len * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_attention_mask.get(), attention_mask.data(), len * sizeof(int), cudaMemcpyHostToDevice);
//     dylog::DynamicLogger().info("copy Input to GPU success");
//
//     Dims output_dims = ctx->getTensorShape("output"); // 应该是 (1, actual_seq_len, 512)
//     int output_size = static_cast<int>(output_dims.d[1] * output_dims.d[2]); // seq_len * 512
//
//     void* d_output_wrapper = nullptr;
//     cudaMalloc(&d_output_wrapper, output_size * sizeof(float));
//     auto d_output = CudaUniquePtr(d_output_wrapper, cudaDeleter);
//
//     void* d_677_wrapper = nullptr;
//     cudaMalloc(&d_677_wrapper, 512 * sizeof(float));
//     auto d_677 = CudaUniquePtr(d_677_wrapper, cudaDeleter);
//
//     void* bindings[] = {
//         d_input_ids.get(),
//         d_token_type_ids.get(),
//         d_attention_mask.get(),
//         d_output.get(),
//         d_677.get()
//     };
//
//     // 执行推理
//     try{
//         ctx->executeV2(bindings);
//         dylog::DynamicLogger().info("executeV2 success for len:{}", len);
//     }catch(const std::exception& e){
//         dylog::DynamicLogger().error("executeV2 error:{}", e.what());
//     }
//     std::vector<float> embedding(512);
//     cudaMemcpy(embedding.data(), d_677.get(), 512 * sizeof(float), cudaMemcpyDeviceToHost);
//     for (int i = 0; i < 20; i++){
//         dylog::DynamicLogger().info("embedding[{}]:{}", i, embedding[i]);
//     }
//     delete ctx;
//     return embedding;
//
// }
