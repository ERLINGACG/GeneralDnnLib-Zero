//
// Created by HP on 2026/5/9.
//

#ifndef GENERALDNNLIB_ZERO_ONNX_RT_LLM_EXPORT_H
#define GENERALDNNLIB_ZERO_ONNX_RT_LLM_EXPORT_H
#include "onnx_rt_llm_data.h"
#include "gdlz/export.h"
#include "gdlz/ort/core/onnx_rt_core_export.h"
#include "gdlz/ort/data/onnx_rt_data.h"

namespace gdlz::ort::llm{
    GDLZ_CORE_API void GetCtxSize();
    GDLZ_CORE_API void GetEngineSize();
    GDLZ_CORE_API void GetKvSize(const data::OnnxRtLLmKv& kv);

    GDLZ_CORE_API void GetLayerNames(ort::data::OnnxRTEngine& engine,data::OnnxRtLLmCtx& ctx);

    GDLZ_CORE_API void CreateLLmEngine(data::OnnxRtLLmEngine& llm_engine,ort::data::OnnxRTEngine& core_engine,
                                       core::OnnxRtFramework* framework);

    GDLZ_CORE_API void InitBatch    (data::OnnxRtLLmEngine& engine,data::OnnxRtLLmCtx& ctx,
                                     data::OnnxRtLLmKv& kv, void* tokens,int64_t len);
    GDLZ_CORE_API void GenerateToken(data::OnnxRtLLmEngine& engine,data::OnnxRtLLmCtx& ctx,
                                     data::OnnxRtLLmKv& kv);

   }
#endif //GENERALDNNLIB_ZERO_ONNX_RT_LLM_EXPORT_H