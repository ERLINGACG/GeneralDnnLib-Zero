//
// Created by HP on 2026/5/15.
//

#include "gdlz/ort/core/onnx_rt_core_director.h"
#include "gdlz/ort/core/onnx_rt_framework.h"
#include "gdlz/ort/llm/onnx_rt_llm_director.h"

#include <iostream>
#include <dylog/logger.h>

void gdlz::ort::llm::OnnxRtLLmDirector::Handle(
    ort::data::OnnxRTEngineInfo& engine_info,
    const char* config_path,nlohmann::json& config
) {
    core::OnnxRtCoreDirector::Handle(engine_info, config_path, config);
}

int gdlz::ort::llm::OnnxRtLLmDirector::CreateEngine(
    data::OnnxRtLLmEngine& engine,
    const ort::data::OnnxRTEngineInfo& engine_info,
    nlohmann::json& config
) {
    try
    {
        engine.head_dim  =  config["head_dim"].get<int>();
        engine.layers    =  config["layers"].get<int>();
        engine.heads     =  config["heads"].get<int>();

        core::OnnxRtFramework::CreateEngine(engine_info, engine.core_engine);
        return 0;
    }catch (std::exception& e) {
        std::cout<<"CreateEngine error "<<e.what()<<std::endl;
        return -1;
    }
}
