//
// Created by HP on 2026/5/3.
//

#ifndef GENERALDNNLIB_ZERO_ONNX_RT_CORE_DIRECTOR_H
#define GENERALDNNLIB_ZERO_ONNX_RT_CORE_DIRECTOR_H
#include "gdlz/ort/data/onnx_rt_data.h"

namespace gdlz::ort::core{
    class OnnxRtCoreDirector{

    public:
        ~OnnxRtCoreDirector() = default;
        static void Handle(data::OnnxRTEngine& engine, const char* config_path);
    };
}
#endif //GENERALDNNLIB_ZERO_ONNX_RT_CORE_DIRECTOR_H