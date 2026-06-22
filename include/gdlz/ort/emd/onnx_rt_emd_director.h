//
// Created by HP on 2026/6/1.
//

#ifndef GENERALDNNLIB_ZERO_ONNX_RT_EMD_DIRECTOR_H
#define GENERALDNNLIB_ZERO_ONNX_RT_EMD_DIRECTOR_H
#include <nlohmann/json.hpp>

#include "onnx_rt_emd_data.h"
#include "gdlz/ort/data/onnx_rt_data.h"
#include "gdlz/export.h"

namespace gdlz::ort::emd{
    class GDLZ_CPP_API OnnxRtEmdDirector {

        public:
            // OnnxRtEmdDirector();
            ~OnnxRtEmdDirector()=default;

            static void Handle(
                ort::data::OnnxRTEngineInfo& engine_info,
                const char* config_path,
                nlohmann::json& config
            );

            static  void CreateEngine(
                data::OnnxRtEmdEngine& engine,
                const ort::data::OnnxRTEngineInfo& engine_info,
                nlohmann::json& config
            );
    };
}
#endif //GENERALDNNLIB_ZERO_ONNX_RT_EMD_DIRECTOR_H