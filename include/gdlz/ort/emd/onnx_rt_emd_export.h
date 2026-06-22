//
// Created by HP on 2026/6/1.
//

#ifndef GENERALDNNLIB_ZERO_ONNX_RT_EMD_EXPORT_H
#define GENERALDNNLIB_ZERO_ONNX_RT_EMD_EXPORT_H
#include "onnx_rt_emd_framework.h"
#include "gdlz/export.h"

namespace gdlz::ort::emd{

    GDLZ_CORE_API OnnxRtEmdFramework* CreateEmdFramework(const char* config_path);

    GDLZ_CORE_API void DestroyEmdFramework(OnnxRtEmdFramework* framework);

    GDLZ_CORE_API void GetEmdFrameworkEngineInfo(OnnxRtEmdFramework* framework);

    GDLZ_CORE_API int SetShape(core::shape::OnnxRtShape& shape,data::OnnxRtEmdCtx& ctx);

    GDLZ_CORE_API int Forward(const OnnxRtEmdFramework* framework,data::OnnxRtEmdCtx& ctx,data::OnnxRtEmdData& data);





}
#endif //GENERALDNNLIB_ZERO_ONNX_RT_EMD_EXPORT_H
