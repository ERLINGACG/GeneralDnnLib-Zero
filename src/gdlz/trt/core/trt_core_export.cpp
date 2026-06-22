//
// Created by HP on 2026/6/5.
//
#include "gdlz/trt/core/trt_core_export.h"

#include "gdlz/except_utils.h"
#include "gdlz/trt/core/trt_core_data.h"
#include "gdlz/trt/core/trt_core_director.h"
#include "gdlz/trt/core/trt_core_framework.h"

void gdlz::trt::core::TestLoad(const char* config_path){
    // TensorRTCoreEngine engine;
    // // TensorRTCoreFramework::CreateEngine(engine,config_path);
    // TensorRTCoreFramework::HotBuildEngine(engine,config_path);
}

void gdlz::trt::core::TestLoad2(const char* config_path)
{
    gdlz::except::try_catch([&](){
        cudaSetDevice(0);
        cudaFree(nullptr);

        TensorRTCoreEngine engine;
        // TensorRTCoreCtx ctx;
        TensorRTCoreDirector::Handle(engine, config_path);
        TensorRTCoreFramework::CreateEngine(engine);
        TensorRTCoreFramework::GetEngineInfo(engine);

        // TensorRTCoreFramework::CreateCtx(engine,ctx);



        return 0;
    });
}

int gdlz::trt::core::InitBuilderEngine(ExportCoreEngine& engine)
{
    engine.engine=std::make_unique<TensorRTCoreEngine>();
    return 0;
}

int gdlz::trt::core::InitBuilderEngineInfo(TensorBuilderEngineInfo& info) {
    return TensorRTCoreFramework::InitShapeInfo(info);
}

int gdlz::trt::core::RegisterShapeInfo(const TensorRtBuilderShapeInfo& shape, TensorBuilderEngineInfo& info) {
    return TensorRTCoreFramework::RegisterShapeInfo(shape,info);
}

int gdlz::trt::core::GetShapeInfoDetail(TensorBuilderEngineInfo& info)
{
    return TensorRTCoreFramework::GetShapeInfoDetail(info);
}

int gdlz::trt::core::BuildEngine(ExportCoreEngine& engine, const char* config_path,
    TensorBuilderEngineInfo& info)
{
    return TensorRTCoreFramework::BuildEngine(engine.getCoreEngine(),config_path,info);
}

