//
// Created by HP on 2026/5/9.
//

#ifndef GENERALDNNLIB_ZERO_ONNX_RT_LLM_EXPORT_H
#define GENERALDNNLIB_ZERO_ONNX_RT_LLM_EXPORT_H
#include "onnx_rt_llm_data.h"
#include "onnx_rt_llm_framework.h"
#include "gdlz/export.h"
#include "gdlz/ort/core/onnx_rt_core_export.h"
#include "gdlz/ort/data/onnx_rt_data.h"

namespace gdlz::ort::llm{


    GDLZ_CORE_API void GetCtxSize();

    GDLZ_CORE_API void GetEngineSize();

    GDLZ_CORE_API void GetKvSize(const data::OnnxRtLLmKv& kv);

    GDLZ_CORE_API void GetCtxInfo(const data::OnnxRtLLmCtx& ctx);


    GDLZ_CORE_API OnnxRtLLmFramework* CreateLLmFramework(const char* config_path);

    GDLZ_CORE_API void GetLLmEngineInfo(OnnxRtLLmFramework* framework);

    GDLZ_CORE_API int GetHeads(OnnxRtLLmFramework* framework);

    GDLZ_CORE_API int GetLayers(OnnxRtLLmFramework* framework);

    GDLZ_CORE_API int GetHeadDim(OnnxRtLLmFramework* framework);

    GDLZ_CORE_API int ResetCtx(data::OnnxRtLLmCtx& ctx);

    GDLZ_CORE_API int ResetKv(data::OnnxRtLLmKv& kv);

    GDLZ_CORE_API int InitSampler(sampler::OnnxRtLLmParam& param, data::OnnxRtLLmCtx& ctx);

    GDLZ_CORE_API int PrefillFor1DRoPE(const OnnxRtLLmFramework* framework,data::OnnxRtLLmCtx& ctx,
                                       data::OnnxRtLLmKv& kv, void* tokens,int64_t len);

    GDLZ_CORE_API int DecodeFor1DRoPE(const OnnxRtLLmFramework* framework,data::OnnxRtLLmCtx& ctx,
                                     data::OnnxRtLLmKv& kv);


    GDLZ_CORE_API int InitBatchForTokenIds(const OnnxRtLLmFramework* framework,data::OnnxRtLLmCtx& ctx,
                                           data::OnnxRtLLmKv& kv, void* tokens,int64_t len);


    GDLZ_CORE_API int GenerateToken (const OnnxRtLLmFramework* framework,data::OnnxRtLLmCtx& ctx,
                                     data::OnnxRtLLmKv& kv);

   }
#endif //GENERALDNNLIB_ZERO_ONNX_RT_LLM_EXPORT_H