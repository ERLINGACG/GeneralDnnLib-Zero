//
// Created by HP on 2026/4/19.
//

#include "gdlz/trt/rag/trt_emb_director.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include "NvInfer.h"

#include <dylog/logger.h>
class SilentLogger : public nvinfer1::ILogger {
    void log(Severity, const char*) noexcept override {}
} gLogger;

gdlz::trt::rag::TrtEmbDirector& gdlz::trt::rag::TrtEmbDirector::Handle(
    TrtEmbEngine& engine, const char* config_Path
    )
{
    using std::ifstream;
    using std::ofstream;
    using std::string;
    using nlohmann::json;
    using nvinfer1::ILogger;
    using namespace nvinfer1;

    dylog::DynamicLogger().info("TrtEmbDirector::Handle");
    json j;
    const auto jsonfile = std::make_unique<ifstream>(config_Path);
    *jsonfile >> j;
    dylog::DynamicLogger().info("TrtEmbDirector::Handle, config: {}", j.dump().c_str());
    dylog::DynamicLogger().info("TrtEmbDirector::Handle, config: {}", j["model_path"].dump().c_str());

    std::string model_path = j["model_path"].get<std::string>();
    std::ifstream f(model_path, std::ios::binary);
    std::string data((std::istreambuf_iterator<char>(f)), {});

    dylog::DynamicLogger().info("TrtEmbDirector::Handle, data size: {}", data.size());

    engine.runtime.reset(createInferRuntime(gLogger));
    engine.engine.reset(engine.runtime->deserializeCudaEngine(data.data(), data.size()));
    if (engine.engine!=nullptr){
        dylog::DynamicLogger().info("TrtEmbDirector::Handle success");
        int nbBindings = engine.engine->getNbIOTensors();
        for (int32_t i = 0; i < nbBindings; ++i){

            const char* tensorName = engine.engine->getIOTensorName(i);
            auto type = engine.engine->getTensorDataType(tensorName);
            TensorIOMode ioMode = engine.engine->getTensorIOMode(tensorName);
            Dims dims = engine.engine->getTensorShape(tensorName);

            std::string shapeStr;
            for (int d = 0; d < dims.nbDims; ++d) {
                shapeStr += std::to_string(dims.d[d]);
                if (d != dims.nbDims - 1) shapeStr += ", ";
            }
            const char* ioStr = (ioMode == TensorIOMode::kINPUT) ? "INPUT" : "OUTPUT";
            dylog::DynamicLogger().info("TrtEmbDirector::Handle, tensorName: {}, ioMode: {}, shape: {}", tensorName, ioStr, shapeStr.c_str());
            if (type==DataType::kINT64){
                dylog::DynamicLogger().info("tensorName: {} is int64", tensorName);
            }
            else if (type==DataType::kFLOAT){
                dylog::DynamicLogger().info("tensorName: {} is float32", tensorName);
            }

            engine.engine_info.push_back({tensorName, shapeStr});
        }
    }else{
        dylog::DynamicLogger().error("TrtEmbDirector::Handle failed");
    }
    return *this;
}
