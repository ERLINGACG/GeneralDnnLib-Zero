//
// Created by HP on 2026/5/9.
//

#ifndef GENERALDNNLIB_ZERO_ONNX_RT_LLM_FRAMEWORK_H
#define GENERALDNNLIB_ZERO_ONNX_RT_LLM_FRAMEWORK_H
#include "onnx_rt_llm_data.h"
#include "gdlz/ort/data/onnx_rt_data.h"
#include "gdlz/export.h"
#include "gdlz/ort/core/onnx_rt_core_director.h"

namespace gdlz::ort::llm {

    class OnnxRtLLmSampler {
        public:
            OnnxRtLLmSampler() = default;
            ~OnnxRtLLmSampler() = default;
    };


    class GDLZ_CPP_API  OnnxRtLLmFramework {

        public:
            OnnxRtLLmFramework() = default;
            ~OnnxRtLLmFramework() = default;

            static int GetLayerNames(
                ort::data::OnnxRTEngine& engine,
                data::OnnxRtLLmCtx& ctx
            );

            static int CreateLLmEngine(
                data::OnnxRtLLmEngine& LlmEngine,
                ort::data::OnnxRTEngine& CoreEngine,
                const ort::data::OnnxRTEngineInfo& engine_info
            );



            static int InitBatch(
                data::OnnxRtLLmEngine& engine,
                data::OnnxRtLLmCtx& ctx,
                data::OnnxRtLLmKv& kv,
                void* tokens,int len
            );

            static int GenerateToken(
                data::OnnxRtLLmEngine& engine,
                data::OnnxRtLLmCtx& ctx,
                data::OnnxRtLLmKv& kv
            );

    };
}
#endif //GENERALDNNLIB_ZERO_ONNX_RT_LLM_FRAMEWORK_H