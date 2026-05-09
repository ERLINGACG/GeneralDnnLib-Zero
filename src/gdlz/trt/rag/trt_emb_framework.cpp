//
// Created by HP on 2026/4/19.
//
#include "gdlz/trt/rag/trt_emb_framework.h"

#include <iostream>
#include <ostream>
#include <dylog/logger.h>
#include <cuda_fp16.h>   // 提供 half / __half 类型
#include <cuda_runtime.h>
#include <ranges>
#include <unordered_map>

namespace {
    int CopyDeviceTensorToFloatHost(
        const void* device_ptr,
        int64_t len,
        nvinfer1::DataType data_type,
        cudaStream_t stream,
        std::vector<float>& host_data
    ) {
        if (device_ptr == nullptr) {
            dylog::DynamicLogger().error("device pointer is null");
            return -1;
        }

        host_data.resize(static_cast<size_t>(len));
        if (data_type == nvinfer1::DataType::kFLOAT) {
            CHECK_CUDA(cudaMemcpyAsync(
                host_data.data(),
                device_ptr,
                static_cast<size_t>(len) * sizeof(float),
                cudaMemcpyDeviceToHost
                ,stream
            ));
            return 0;
        }

        if (data_type == nvinfer1::DataType::kHALF) {
            std::vector<half> half_data(static_cast<size_t>(len));
            CHECK_CUDA(cudaMemcpyAsync(
                half_data.data(),
                device_ptr,
                static_cast<size_t>(len) * sizeof(half),
                cudaMemcpyDeviceToHost
                ,stream
            ));

            for (int64_t i = 0; i < len; ++i) {
                host_data[static_cast<size_t>(i)] = __half2float(half_data[static_cast<size_t>(i)]);
            }
            return 0;
        }

        dylog::DynamicLogger().error("unsupported output data type: {}", static_cast<int>(data_type));
        return -1;
    }

    int CopyDeviceTensorToInt64Host(
        const void* device_ptr,
        int64_t len,
        nvinfer1::DataType data_type,
        cudaStream_t stream,
        std::vector<int64_t>& host_data
    ) {
        if (device_ptr == nullptr) {
            dylog::DynamicLogger().error("device pointer is null");
            return -1;
        }

        host_data.resize(static_cast<size_t>(len));
        if (data_type == nvinfer1::DataType::kINT64) {
            CHECK_CUDA(cudaMemcpyAsync(
                host_data.data(),
                device_ptr,
                static_cast<size_t>(len) * sizeof(int64_t),
                cudaMemcpyDeviceToHost
                ,stream
            ));
            return 0;
        }

        if (data_type == nvinfer1::DataType::kINT32) {
            std::vector<int32_t> int32_data(static_cast<size_t>(len));
            CHECK_CUDA(cudaMemcpyAsync(
                int32_data.data(),
                device_ptr,
                static_cast<size_t>(len) * sizeof(int32_t),
                cudaMemcpyDeviceToHost,
                stream
            ));

            for (int64_t i = 0; i < len; ++i) {
                host_data[static_cast<size_t>(i)] = int32_data[static_cast<size_t>(i)];
            }
            return 0;
        }

        dylog::DynamicLogger().error("unsupported mask data type: {}", static_cast<int>(data_type));
        return -1;
    }
}
// 生成 TRT 引擎
// trtexec --onnx=model_1.onnx --minShapes=input_ids:1x1,token_type_ids:1x1,attention_mask:1x1 --optShapes=input_ids:1x128,token_type_ids:1x128,attention_mask:1x128 --maxShapes=input_ids:1x512,token_type_ids:1x512,attention_mask:1x512 --saveEngine=model_1.trt
void gdlz::trt::rag::TrtEmbFramework::ResourceDirector(
    TrtEmbDirector& director,
    const char* config_path
)
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
    using namespace nvinfer1;
    ctx.reset();
    ctx.context.reset(engine.engine->createExecutionContext());
    if (ctx.context==nullptr)
    {
        dylog::DynamicLogger().error("create ExecutionContext failed");
        return -1;
    }

    ctx.stream = new cudaStream_t();
    // auto startime = std::chrono::high_resolution_clock::now();
    auto status = cudaStreamCreate(ctx.stream);
    // auto endtime = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endtime - startime);
    // std::cout << "cudaStreamCreate adder " << ctx.stream << " " << status << " " << duration.count() / 1000.0 << std::endl;

    if (status != cudaSuccess) {
        dylog::DynamicLogger().error("create cuda stream failed: {}", cudaGetErrorString(status));
        delete ctx.stream;
        ctx.stream = nullptr;
        ctx.context.reset();
        return -1;
    }

    if (ctx.bindings == nullptr) {
        ctx.bindings = std::make_unique<std::map<std::string, void*>>();
    } else {
        ctx.bindings->clear();
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

    if (ctx.context == nullptr || ctx.bindings == nullptr || ctx.stream == nullptr) {
        dylog::DynamicLogger().error("context or bindings or stream is null");
        return -1;
    }

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
    if (it->first == DataType::kINT64)
    {
        dylog::DynamicLogger().info("is INT64");
    }
    if (binding->second != nullptr) {
        auto status = cudaFree(binding->second);
        if (status != cudaSuccess) {
            dylog::DynamicLogger().error("free old binding failed, tensorName: {}, error: {}", name, cudaGetErrorString(status));
            return -1;
        }
        binding->second = nullptr;
    }

    if (isMask==2){

        CHECK_CUDA(
            cudaMalloc(&binding->second, len * put_size)
            );
        if (it->first == DataType::kINT64) {
            const std::vector<int64_t> mask(len, 1);
            CHECK_CUDA(
                cudaMemcpyAsync(binding->second, mask.data(), len * sizeof(int64_t),
                        cudaMemcpyHostToDevice, *ctx.stream
                        )
            );
        } else if (it->first == DataType::kINT32) {
            const std::vector<int32_t> mask(len, 1);
            CHECK_CUDA(
                cudaMemcpyAsync(binding->second, mask.data(), len * sizeof(int32_t),
                    cudaMemcpyHostToDevice, *ctx.stream
                    )
            );
        } else {
            dylog::DynamicLogger().error("tensorName: {} mask type unsupported", name);
            return -1;
        }
        dylog::DynamicLogger().info("tensorName: {}, shape: {} ,all 1", name,len);


    }else if (isMask==1){

        CHECK_CUDA(
            cudaMalloc(&binding->second, len * put_size)
            );
        if (it->first == DataType::kINT64) {
            const std::vector<int64_t> mask(len, 0);
            CHECK_CUDA(
                cudaMemcpyAsync(binding->second, mask.data(), len * sizeof(int64_t),
                        cudaMemcpyHostToDevice, *ctx.stream
                        )
            );
        } else if (it->first == DataType::kINT32) {
            const std::vector mask(len, 0);
            CHECK_CUDA(
                cudaMemcpyAsync(binding->second, mask.data(), len * sizeof(int32_t),
                    cudaMemcpyHostToDevice, *ctx.stream
                    )
            );
        } else {
            dylog::DynamicLogger().error("tensorName: {} mask type unsupported", name);
            return -1;
        }
        dylog::DynamicLogger().info("tensorName: {}, shape: {} ,all 0", name,len);
    }else{
        CHECK_CUDA(
            cudaMalloc(&binding->second, len * put_size)
            );
        CHECK_CUDA(
            cudaMemcpyAsync(binding->second, input, len * put_size,
                cudaMemcpyHostToDevice, *ctx.stream
            )
        );
    }

    return 0;
}

int gdlz::trt::rag::TrtEmbFramework::SetDims(
    TrtEmbCtx& ctx,
    const char* name,
    const int* input, int len
)
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

int gdlz::trt::rag::TrtEmbFramework::ClearBindings(TrtEmbCtx& ctx)
{
    if (ctx.bindings == nullptr) {
        return 0;
    }

    for (auto& [name, binding] : *ctx.bindings) {
        if (binding == nullptr) {
            continue;
        }

        auto status = cudaFree(binding);
        if (status != cudaSuccess) {
            dylog::DynamicLogger().error(
                "clear binding failed, tensorName: {}, error: {}",
                name,
                cudaGetErrorString(status)
            );
            return -1;
        }
        binding = nullptr;
    }

    ctx.bindings->clear();
    return 0;
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
    if (ctx.bindings == nullptr) {
        ctx.bindings = std::make_unique<std::map<std::string, void*>>();
    }
    if (!ctx.bindings->contains(name)) {
        ctx.bindings->insert({name, nullptr});
    }

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
    if (binding->second != nullptr) {
        auto status = cudaFree(binding->second);
        if (status != cudaSuccess) {
            dylog::DynamicLogger().error("free old binding failed, tensorName: {}, error: {}", name, cudaGetErrorString(status));
            return -1;
        }
        binding->second = nullptr;
    }
    CHECK_CUDA(cudaMalloc(&binding->second, size * type_size->second));

    if (!ctx.context->setTensorAddress(name, binding->second)) {
        dylog::DynamicLogger().error("set tensor address failed, tensorName: {}", name);
        return -1;
    }

    return 0;

}

int gdlz::trt::rag::TrtEmbFramework::ForwardAsync(TrtEmbCtx& ctx) const
{
    if (ctx.context == nullptr || ctx.bindings == nullptr || ctx.stream == nullptr) {
        dylog::DynamicLogger().error("context or bindings or stream is null");
        return -1;
    }

    for (const auto& key : engine.engine_info | std::views::keys){
        auto binding = ctx.bindings->find(key);
        if (binding == ctx.bindings->end() || binding->second == nullptr) {
            dylog::DynamicLogger().error("tensorName: {} binding not ready", key);
            return -1;
        }

        if (!ctx.context->setTensorAddress(key.c_str(), binding->second)) {
            dylog::DynamicLogger().error("set tensor address failed, tensorName: {}", key);
            return -1;
        }
    }

    try{
        auto res=ctx.context->enqueueV3(*ctx.stream);
        if (!res){
            dylog::DynamicLogger().error("enqueueV3 failed");
            return -1;
        }
        return 0;
    }catch (const std::exception& e){
        dylog::DynamicLogger().error("enqueueV3 failed: {}", e.what());
        return -1;
    }
}

int gdlz::trt::rag::TrtEmbFramework::Synchronize(const TrtEmbCtx& ctx)
{
    if (ctx.stream == nullptr) {
        dylog::DynamicLogger().error("stream is null");
        return -1;
    }

    auto status = cudaStreamSynchronize(*ctx.stream);
    if (status != cudaSuccess) {
        dylog::DynamicLogger().error("stream synchronize failed: {}", cudaGetErrorString(status));
        return -1;
    }
    return 0;
}

int gdlz::trt::rag::TrtEmbFramework::Forward(TrtEmbCtx& ctx) const
{
    if (ForwardAsync(ctx) != 0) {
        return -1;
    }

    return Synchronize(ctx);
}

int gdlz::trt::rag::TrtEmbFramework::GetEmbedding(
    const TrtEmbCtx& ctx,
    TrtEmbData& embedding,
    const char* name
) const {

    if (ctx.context == nullptr || ctx.bindings == nullptr || ctx.stream == nullptr) {
        dylog::DynamicLogger().error("context or bindings or stream is null");
        return -1;
    }

    if (Synchronize(ctx) != 0) {
        return -1;
    }

    auto type = this->engine.engine->getTensorDataType(name);
    auto dims = ctx.context->getTensorShape(name);
    auto type_size = typeSizeMap.find(type);
    if (type_size == typeSizeMap.end()) {
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
    auto copy_status = cudaMemcpy(vec.data(), binding->second, vec.size() * type_size->second, cudaMemcpyDeviceToHost);
    if (copy_status != cudaSuccess) {
        dylog::DynamicLogger().error("copy embedding to host failed, tensorName: {}, error: {}", name, cudaGetErrorString(copy_status));
        return -1;
    }
    embedding.embedding = std::make_unique<float[]>(vec.size());
    embedding.embedding_size = static_cast<int>(vec.size());
    std::ranges::copy(vec, embedding.embedding.get());
    return 0;
}

int gdlz::trt::rag::TrtEmbFramework::GetPooledEmbedding(
    const TrtEmbCtx& ctx,
    TrtEmbData& embedding,
    const char* hidden_state_name,
    const char* attention_mask_name
) const {
    if (ctx.context == nullptr || ctx.bindings == nullptr || ctx.stream == nullptr) {
        dylog::DynamicLogger().error("context or bindings or stream is null");
        return -1;
    }

    if (Synchronize(ctx) != 0) {
        return -1;
    }

    auto hidden_binding = ctx.bindings->find(hidden_state_name);
    if (hidden_binding == ctx.bindings->end()) {
        dylog::DynamicLogger().error("tensorName: {} not found", hidden_state_name);
        return -1;
    }

    auto mask_binding = ctx.bindings->find(attention_mask_name);
    if (mask_binding == ctx.bindings->end()) {
        dylog::DynamicLogger().error("tensorName: {} not found", attention_mask_name);
        return -1;
    }

    const auto hidden_dims = ctx.context->getTensorShape(hidden_state_name);
    const auto mask_dims = ctx.context->getTensorShape(attention_mask_name);
    if (hidden_dims.nbDims != 3) {
        dylog::DynamicLogger().error(
            "tensorName: {} invalid dims count: {}, expected 3",
            hidden_state_name,
            hidden_dims.nbDims
        );
        return -1;
    }
    if (mask_dims.nbDims != 2) {
        dylog::DynamicLogger().error(
            "tensorName: {} invalid dims count: {}, expected 2",
            attention_mask_name,
            mask_dims.nbDims
        );
        return -1;
    }

    const int batch_size = hidden_dims.d[0];
    const int sequence_length = hidden_dims.d[1];
    const int hidden_size = hidden_dims.d[2];
    if (batch_size <= 0 || sequence_length <= 0 || hidden_size <= 0) {
        dylog::DynamicLogger().error(
            "tensorName: {} invalid shape: [{}, {}, {}]",
            hidden_state_name,
            batch_size,
            sequence_length,
            hidden_size
        );
        return -1;
    }

    if (mask_dims.d[0] != batch_size || mask_dims.d[1] != sequence_length) {
        dylog::DynamicLogger().error(
            "shape mismatch, {}: [{}, {}, {}], {}: [{}, {}]",
            hidden_state_name,
            batch_size,
            sequence_length,
            hidden_size,
            attention_mask_name,
            mask_dims.d[0],
            mask_dims.d[1]
        );
        return -1;
    }

    std::vector<float> hidden_host;
    if (CopyDeviceTensorToFloatHost(
        hidden_binding->second,
        static_cast<int64_t>(batch_size) * sequence_length * hidden_size,
        this->engine.engine->getTensorDataType(hidden_state_name),
        *ctx.stream,
        hidden_host
    ) != 0) {
        return -1;
    }

    std::vector<int64_t> mask_host;
    if (CopyDeviceTensorToInt64Host(
        mask_binding->second,
        static_cast<int64_t>(batch_size) * sequence_length,
        this->engine.engine->getTensorDataType(attention_mask_name),
        *ctx.stream,
        mask_host
    ) != 0) {
        return -1;
    }

    std::vector<float> pooled(static_cast<size_t>(batch_size) * hidden_size, 0.0f);
    for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
        int64_t valid_token_count = 0;
        for (int seq_index = 0; seq_index < sequence_length; ++seq_index) {
            const auto mask_value = mask_host[static_cast<size_t>(batch_index) * sequence_length + seq_index];
            if (mask_value == 0) {
                continue;
            }

            ++valid_token_count;
            for (int hidden_index = 0; hidden_index < hidden_size; ++hidden_index) {
                const auto hidden_offset =
                    (static_cast<size_t>(batch_index) * sequence_length + seq_index) * hidden_size + hidden_index;
                const auto pooled_offset = static_cast<size_t>(batch_index) * hidden_size + hidden_index;
                pooled[pooled_offset] += hidden_host[hidden_offset];
            }
        }

        if (valid_token_count == 0) {
            dylog::DynamicLogger().warn(
                "attention_mask valid token count is 0 at batch index: {}",
                batch_index
            );
            continue;
        }

        const float scale = 1.0f / static_cast<float>(valid_token_count);
        for (int hidden_index = 0; hidden_index < hidden_size; ++hidden_index) {
            const auto pooled_offset = static_cast<size_t>(batch_index) * hidden_size + hidden_index;
            pooled[pooled_offset] *= scale;
        }
    }

    embedding.embedding = std::make_unique<float[]>(pooled.size());
    embedding.embedding_size = static_cast<int>(pooled.size());
    std::ranges::copy(pooled, embedding.embedding.get());
    return 0;
}

void gdlz::trt::rag::TrtEmbFramework::DestroyRt() {
    engine.engine.reset();
}

