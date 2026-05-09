//
// Created by HP on 2026/5/3.
//
#include "gdlz/ort/core/onnx_rt_core_director.h"

#include <fstream>
#include <iostream>
#include <dylog/logger.h>

#include "nlohmann/json.hpp"
void gdlz::ort::core::OnnxRtCoreDirector::Handle(data::OnnxRTEngine& engine, const char* config_path)
{
    using nlohmann::json;

    json j;
    const auto jsonfile = std::make_unique<std::ifstream>(config_path);

    if (!jsonfile->is_open())
    {
        std::cerr<<"Error opening file "<<config_path<<std::endl;
        return;
    }*jsonfile >> j;


    // std::string model_path = j["model_path"].get<std::string>();
    // bool use_cuda = config["use_cuda"].get<bool>();
    engine.model_path = j["model_path"].get<std::string>();
    if (j["use_cuda"].get<bool>())
    {
        engine.use_cuda =true;
        dylog::DynamicLogger().setInvokeName("OnnxRtCoreDirector::Handle").info("OnnxRtCoreDirector::Handle, use_cuda: {}", engine.use_cuda);
    }
    engine.env=Ort::Env();
}
