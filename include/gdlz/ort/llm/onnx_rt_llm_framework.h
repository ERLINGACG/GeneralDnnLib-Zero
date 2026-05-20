//
// Created by HP on 2026/5/9.
//

#ifndef GENERALDNNLIB_ZERO_ONNX_RT_LLM_FRAMEWORK_H
#define GENERALDNNLIB_ZERO_ONNX_RT_LLM_FRAMEWORK_H
#include <random>

#include "onnx_rt_llm_data.h"
#include "gdlz/ort/data/onnx_rt_data.h"
#include "gdlz/export.h"
#include "gdlz/ort/core/onnx_rt_core_director.h"

namespace gdlz::ort::llm {

    class GDLZ_CPP_API  OnnxRtLLmFramework {

        data::OnnxRtLLmEngine engine;
        ort::data::OnnxRTEngineInfo engine_info;
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    public:

            OnnxRtLLmFramework(const char* config_path);
            ~OnnxRtLLmFramework() = default;

            void GetLayers();

            static int InitSampler(
                sampler::OnnxRtLLmParam& param,
                data::OnnxRtLLmCtx& ctx
            );

            int SetInput(
                ort::data::OnnxRtShape& shape,
                ort::data::OnnxRtInput& input
            ) const;



            int Prefill(
                data::OnnxRtLLmCtx& ctx,
                data::OnnxRtLLmKv& kv,
                const ort::data::OnnxRtInput& input
            ) const;

            int Decode(
                data::OnnxRtLLmCtx& ctx,
                data::OnnxRtLLmKv& kv,
                ort::data::OnnxRtInput& input
            ) const;

            int InitBatchForTokenIds(
                data::OnnxRtLLmCtx& ctx,
                data::OnnxRtLLmKv& kv,
                void* tokens,int64_t len
            ) const;

            int GenerateToken(
                data::OnnxRtLLmCtx& ctx,
                data::OnnxRtLLmKv& kv
            ) const;

    };
}
#endif //GENERALDNNLIB_ZERO_ONNX_RT_LLM_FRAMEWORK_H