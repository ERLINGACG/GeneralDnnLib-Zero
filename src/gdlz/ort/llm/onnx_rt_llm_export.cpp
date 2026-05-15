//
// Created by HP on 2026/5/9.
//
#include "gdlz/ort/llm/onnx_rt_llm_export.h"

#include <iostream>

#include "gdlz/ort/core/onnx_rt_framework.h"
#include "gdlz/ort/llm/onnx_rt_llm_data.h"
#include "gdlz/ort/llm/onnx_rt_llm_framework.h"

void gdlz::ort::llm::GetCtxSize() {
    std::cout << sizeof(data::OnnxRtLLmCtx) << std::endl;
    std::cout << "offsetof(input_names_ptr) = "  << offsetof(data::OnnxRtLLmCtx, input_names_ptr) << std::endl;
    std::cout << "offsetof(output_names_ptr) = " << offsetof(data::OnnxRtLLmCtx, output_names_ptr) << std::endl;
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

void gdlz::ort::llm::GetLayerNames(ort::data::OnnxRTEngine& engine, data::OnnxRtLLmCtx& ctx)
{
     OnnxRtLLmFramework::GetLayerNames(engine, ctx);
}

void gdlz::ort::llm::CreateLLmEngine(data::OnnxRtLLmEngine& llm_engine, ort::data::OnnxRTEngine& core_engine,
    core::OnnxRtFramework* framework)
{
    OnnxRtLLmFramework::CreateLLmEngine(llm_engine, core_engine, framework->GetEngineInfo_());
}



void gdlz::ort::llm::InitBatch(data::OnnxRtLLmEngine& engine, data::OnnxRtLLmCtx& ctx,
                               data::OnnxRtLLmKv& kv, void* tokens, int64_t len){

    OnnxRtLLmFramework::InitBatch(engine, ctx, kv, tokens, len);

}

void gdlz::ort::llm::GenerateToken(data::OnnxRtLLmEngine& engine, data::OnnxRtLLmCtx& ctx,
    data::OnnxRtLLmKv& kv)
{
    OnnxRtLLmFramework::GenerateToken(engine, ctx, kv);
}
