//
// Created by HP on 2026/5/9.
//
#include "gdlz/ort/llm/onnx_rt_llm_export.h"

#include <iostream>

#include "gdlz/except_utils.h"
#include "gdlz/ort/llm/onnx_rt_llm_data.h"
#include "gdlz/ort/llm/onnx_rt_llm_framework.h"

void gdlz::ort::llm::GetCtxSize() {
    std::cout << sizeof(data::OnnxRtLLmCtx) << std::endl;
    std::cout << "offsetof(batch) = "            << offsetof(data::OnnxRtLLmCtx, batch) << std::endl;
    std::cout << "offsetof(token_pos) = "        << offsetof(data::OnnxRtLLmCtx, token_pos) << std::endl;
    std::cout << "offsetof(next_token_id) = "    << offsetof(data::OnnxRtLLmCtx, next_token_id) << std::endl;
}

void gdlz::ort::llm::GetEngineSize()
{
    std::cout << sizeof(data::OnnxRtLLmEngine) << std::endl;
    std::cout << "offsetof(core_engine) = " << offsetof(data::OnnxRtLLmEngine, core_engine) << std::endl;
    std::cout << "offsetof(head_dim) =     " << offsetof(data::OnnxRtLLmEngine, head_dim) << std::endl;
    std::cout << "offsetof(heads) =        " << offsetof(data::OnnxRtLLmEngine, heads) << std::endl;
    std::cout << "offsetof(layers) =       " << offsetof(data::OnnxRtLLmEngine, layers) << std::endl;
}

void gdlz::ort::llm::GetKvSize(const data::OnnxRtLLmKv& kv)
{
    std::cout << "kv size = " << kv.kv->size() << std::endl;
}


void gdlz::ort::llm::GetCtxInfo(const data::OnnxRtLLmCtx& ctx)
{
    ctx.info();
}

gdlz::ort::llm::OnnxRtLLmFramework* gdlz::ort::llm::CreateLLmFramework(const char* config_path) {

    return new OnnxRtLLmFramework(config_path);
}

void gdlz::ort::llm::GetLLmEngineInfo(OnnxRtLLmFramework* framework)
{
    framework->GetLayerInfo();
}

int gdlz::ort::llm::GetHeads(OnnxRtLLmFramework* framework)
{
    return framework->GetHeads();
}

int gdlz::ort::llm::GetLayers(OnnxRtLLmFramework* framework)
{
    return framework->GetLayers();
}

int gdlz::ort::llm::GetHeadDim(OnnxRtLLmFramework* framework)
{
    return framework->GetHeadDim();
}

int gdlz::ort::llm::ResetCtx(data::OnnxRtLLmCtx& ctx)
{
    return gdlz::except::try_catch<>([&](){
        ctx.reset();
        return 0;
    });
}

int gdlz::ort::llm::ResetKv(data::OnnxRtLLmKv& kv)
{
    return gdlz::except::try_catch<>([&]() {
        kv.reset();
        return 0;
    });
}


int gdlz::ort::llm::InitSampler(sampler::OnnxRtLLmParam& param, data::OnnxRtLLmCtx& ctx)
{
   return  OnnxRtLLmFramework::InitSampler(param, ctx);
}

int gdlz::ort::llm::PrefillFor1DRoPE(const OnnxRtLLmFramework* framework, data::OnnxRtLLmCtx& ctx,
    data::OnnxRtLLmKv& kv, void* tokens, int64_t len)
{
    return gdlz::except::try_catch<>([&](){

            return framework->PrefillFor1DRoPE(ctx, kv, tokens, len);
        }
    );
}

int gdlz::ort::llm::DecodeFor1DRoPE(const OnnxRtLLmFramework* framework, data::OnnxRtLLmCtx& ctx,
    data::OnnxRtLLmKv& kv)
{
    return gdlz::except::try_catch<>([&](){

            return framework->DecodeFor1DRoPE(ctx, kv);
        }
    );
}


int gdlz::ort::llm::InitBatchForTokenIds(const OnnxRtLLmFramework* framework, data::OnnxRtLLmCtx& ctx,
                                         data::OnnxRtLLmKv& kv, void* tokens, int64_t len)
{
    return gdlz::except::try_catch<>(
            [&](){
                return framework->InitBatchForTokenIds(ctx, kv, tokens, len);
            }
        );
}

int gdlz::ort::llm::GenerateToken(const OnnxRtLLmFramework* framework, data::OnnxRtLLmCtx& ctx,
    data::OnnxRtLLmKv& kv) {
    return gdlz::except::try_catch<>(
        [&]() {
            return framework->GenerateToken(ctx, kv);
        }
    );
}


