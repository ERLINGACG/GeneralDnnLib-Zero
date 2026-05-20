//
// Created by HP on 2026/5/3.
//

#ifndef GENERALDNNLIB_ZERO_ONNX_RT_CORE_DIRECTOR_H
#define GENERALDNNLIB_ZERO_ONNX_RT_CORE_DIRECTOR_H
#include "gdlz/ort/data/onnx_rt_data.h"
#include "gdlz/export.h"
#include <nlohmann/json.hpp>
namespace gdlz::ort::core{
    class GDLZ_CPP_API OnnxRtCoreDirector{

    public:
        ~OnnxRtCoreDirector() = default;
        static void Handle(
            data::OnnxRTEngineInfo& engine_info, const char* config_path,nlohmann::json& config
        );
    };
}
#endif //GENERALDNNLIB_ZERO_ONNX_RT_CORE_DIRECTOR_H