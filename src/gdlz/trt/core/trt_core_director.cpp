//
// Created by HP on 2026/6/5.
//
#include "gdlz/trt/core/trt_core_director.h"

#include <fstream>
#include <dylog/logger.h>

int gdlz::trt::core::TensorRTCoreDirector::Handle(TensorRTCoreEngine& engine,const char* config_path)
{

    auto load_file=[&](){
        const auto jsonfile = std::make_unique<std::ifstream>(config_path);
        if (!jsonfile->is_open()) {
            return -1;
        }
        *jsonfile >> engine.config;
        dylog::DynamicLogger().debug("TensorRTCoreDirector::Handle, config: {}", engine.config.dump().c_str());
        return 0;
    };

    auto load_env=[&](){
        if (!engine.config.contains("env")) {
            dylog::DynamicLogger().error("TensorRTCoreDirector::Handle, config: env is missing");
            return -1;
        }
        engine.env = engine.config[engine.config["env"].get<std::string>()];
        dylog::DynamicLogger().debug("TensorRTCoreDirector::Handle, env: {}", engine.env.dump().c_str());
        if (!engine.env.contains("model_path")) {
            dylog::DynamicLogger().error("TensorRTCoreDirector::Handle, env: model_path is missing");
            return -1;
        }
        std::string model_path = engine.env["model_path"].get<std::string>();
        dylog::DynamicLogger().debug("TensorRTCoreDirector::Handle, model_path: {}", model_path.c_str());
        return 0;
    };



    int ret = load_file();
    if (ret!=0) {
        return ret;
    }
    ret = load_env();
    if (ret!=0) {
        return ret;
    }

    return 0;
}

int gdlz::trt::core::TensorRTCoreDirector::OnnxHandler(TensorRTCoreEngine& engine, const char* config_path)
{

    return 0;
}
