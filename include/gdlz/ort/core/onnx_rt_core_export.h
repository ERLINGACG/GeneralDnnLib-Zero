//
// Created by HP on 2026/5/3.
//

#ifndef GENERALDNNLIB_ZERO_ONNX_RT_CORE_EXPORT_H
#define GENERALDNNLIB_ZERO_ONNX_RT_CORE_EXPORT_H
#include "onnx_rt_framework.h"
#include "gdlz/export.h"

namespace gdlz::ort::core{

    GDLZ_CORE_API OnnxRtFramework* CreateOnnxRtFramework(const char* config_path);
    GDLZ_CORE_API int CreateEngine(const OnnxRtFramework* framework, data::OnnxRTEngine& ctx);
    GDLZ_CORE_API void GetEngineInfo(const OnnxRtFramework* framework, data::OnnxRTEngine& ctx);
    GDLZ_CORE_API void Run(const OnnxRtFramework* framework, data::OnnxRTEngine& ctx);
    GDLZ_CORE_API int RunExam(const OnnxRtFramework* framework,void* token, int len,data::OnnxRTEngine& ctx);
}
#endif //GENERALDNNLIB_ZERO_ONNX_RT_CORE_EXPORT_H