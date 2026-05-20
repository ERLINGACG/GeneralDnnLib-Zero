//
// Created by HP on 2026/5/15.
//

#ifndef GENERALDNNLIB_ZERO_ONNX_RT_DIRECTOR_H
#define GENERALDNNLIB_ZERO_ONNX_RT_DIRECTOR_H
#include <nlohmann/json_fwd.hpp>

#include "onnx_rt_llm_data.h"
#include "gdlz/ort/core/onnx_rt_core_director.h"
#include "gdlz/ort/data/onnx_rt_data.h"

namespace gdlz::ort::llm{
    class OnnxRtLLmDirector {

        public:
            ~OnnxRtLLmDirector() = default;
            static void Handle(ort::data::OnnxRTEngineInfo& engine_info, const char* config_path,nlohmann::json& config);

            static int  CreateEngine(
                data::OnnxRtLLmEngine& engine,
                const ort::data::OnnxRTEngineInfo& engine_info,
                nlohmann::json& config
            );
    };
}
#endif //GENERALDNNLIB_ZERO_ONNX_RT_DIRECTOR_H