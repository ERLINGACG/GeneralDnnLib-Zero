//
// Created by HP on 2026/6/5.
//

#ifndef GENERALDNNLIB_ZERO_TRT_CORE_EXPORT_H
#define GENERALDNNLIB_ZERO_TRT_CORE_EXPORT_H
#include <string>

#include "trt_core_data.h"
#include "gdlz/export.h"
namespace gdlz::trt::core{
    GDLZ_CORE_API void TestLoad(const char* config_path);

    GDLZ_CORE_API void TestLoad2(const char* config_path);

    GDLZ_CORE_API void TestLoad3(const char* config_path);

    GDLZ_CORE_API int InitBuilderEngine(ExportCoreEngine& engine);

    GDLZ_CORE_API int InitBuilderEngineInfo(TensorBuilderEngineInfo& info);

    GDLZ_CORE_API int RegisterShapeInfo(const TensorRtBuilderShapeInfo& shape,TensorBuilderEngineInfo& info);

    GDLZ_CORE_API int GetShapeInfoDetail(TensorBuilderEngineInfo& info);

    GDLZ_CORE_API int BuildEngine(ExportCoreEngine& engine,const char* config_path,TensorBuilderEngineInfo& info);
}
#endif //GENERALDNNLIB_ZERO_TRT_CORE_EXPORT_H