//
// Created by HP on 2026/6/1.
//
#include "gdlz/ort/emd/onnx_rt_emd_director.h"

#include <dylog/logger.h>

#include "gdlz/ort/core/onnx_rt_core_director.h"
#include "gdlz/ort/core/onnx_rt_framework.h"


void gdlz::ort::emd::OnnxRtEmdDirector::Handle(
    ort::data::OnnxRTEngineInfo& engine_info,
    const char* config_path, nlohmann::json& config
){
    core::OnnxRtCoreDirector::Handle(engine_info, config_path, config);
}

void gdlz::ort::emd::OnnxRtEmdDirector::CreateEngine(data::OnnxRtEmdEngine& engine,
    const ort::data::OnnxRTEngineInfo& engine_info, nlohmann::json& config)
{
    try
    {
        core::OnnxRtFramework::CreateEngine(engine_info, engine.core_engine);

    }catch (std::exception& e) {
        dylog::DynamicLogger().
               setInvokeName("OnnxRtEmdDirector::CreateEngine").
               error("Error creating engine: {}", e.what());
    }
}
