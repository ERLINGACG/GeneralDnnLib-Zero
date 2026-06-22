//
// Created by HP on 2026/6/5.
//
#include "gdlz/trt/core/trt_core_framework.h"

#include <fstream>
#include <iostream>
#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <dylog/logger.h>

#include "gdlz/trt/core/trt_core_director.h"
#include "gdlz/trt/core/trt_code.h"

int gdlz::trt::core::TensorRTCoreFramework::CreateEngine(TensorRTCoreEngine& engine)
{
    using namespace nvinfer1;
    using namespace nvonnxparser;

    if (!engine.env.contains("model_path")){
        return code::config::CONFIG_NOT_FLIED;
    }
    const auto model_path=engine.env["model_path"].get<std::string>();

    std::ifstream f(model_path, std::ios::binary);
    if (!f) { return code::config::READ_CONFIG_FAILED; }

    std::vector data((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());

    dylog::DynamicLogger().info("TrtEmbDirector::Handle, data size: {}", data.size());

    engine.runtime.reset(createInferRuntime(engine.logger));
    engine.engine.reset(engine.runtime->deserializeCudaEngine(data.data(), data.size()));
    if (engine.runtime==nullptr || engine.engine==nullptr){
        return code::operation::RT_FAILURE;
    }
    return code::operation::SUCCESS;
}

int gdlz::trt::core::TensorRTCoreFramework::HotBuildEngine(TensorRTCoreEngine& engine, const char* config_path)
{

    TrtLogger logger;
    using namespace nvinfer1;
    using namespace nvonnxparser;

    int ret = TensorRTCoreDirector::Handle(engine, config_path);
    if (ret != 0) {
        dylog::DynamicLogger().error("TensorRTCoreFramework::HotBuildEngine, Handle failed");
        return ret;
    }
    std::string module_path=engine.env["model_path"].get<std::string>();
    if(!engine.env.contains("engine_path")){
        dylog::DynamicLogger().error("TensorRTCoreFramework::HotBuildEngine, env: engine_path is missing");
        return -1;
    }

    dylog::DynamicLogger().info("TensorRTCoreFramework::HotBuildEngine, engine_path: {}", engine.env["engine_path"].dump().c_str());

    auto build_to_builder=[&]()
    {
        auto builder = std::unique_ptr<IBuilder>(createInferBuilder(logger));

        if (!builder) {
            dylog::DynamicLogger().error("TensorRTCoreFramework::HotBuildEngine, createInferBuilder failed");
            return -1;
        }
        auto flags = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(flags));
        if (!network) {
            dylog::DynamicLogger().error("TensorRTCoreFramework::HotBuildEngine, createNetworkV2 failed");
            return -1;
        }

        auto parser = std::unique_ptr<IParser>(createParser(*network, logger));
        if (!parser) {
            dylog::DynamicLogger().error("TensorRTCoreFramework::HotBuildEngine, createParser failed");
            return -1;
        }

        if (!parser->parseFromFile(module_path.c_str(),
            static_cast<int>(ILogger::Severity::kWARNING))) {
                    dylog::DynamicLogger().error("TensorRTCoreFramework::HotBuildEngine, parseFromFile failed");

                    for (int i = 0; i < parser->getNbErrors(); ++i) {
                        auto* error = parser->getError(i);
                        if (error) {
                            dylog::DynamicLogger().error(
                                "TensorRT ONNX Parser Error[{}]: {}",
                                i,
                                error->desc());
                        }
                    }
            return -1;
        }


        auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
        if (!config) {
            dylog::DynamicLogger().error("TensorRTCoreFramework::HotBuildEngine, createBuilderConfig failed");
            return -1;
        }
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 3ULL << 30);
        config->setFlag(BuilderFlag::kFP16);
        auto profile = builder->createOptimizationProfile();
        if (!profile) {
            dylog::DynamicLogger().error("TensorRTCoreFramework::HotBuildEngine, createOptimizationProfile failed");
            return -1;
        }
        try{
            profile->setDimensions("input_ids", OptProfileSelector::kMIN, Dims2{1, 1});
            profile->setDimensions("input_ids", OptProfileSelector::kOPT, Dims2{1, 512});    // batch=4, cur_len=512
            profile->setDimensions("input_ids", OptProfileSelector::kMAX, Dims2{1, 2048});   // batch=8, cur_len=2048

            profile->setDimensions("attention_mask", OptProfileSelector::kMIN, Dims2{1, 1});
            profile->setDimensions("attention_mask", OptProfileSelector::kOPT, Dims2{1, 512});
            profile->setDimensions("attention_mask", OptProfileSelector::kMAX, Dims2{1, 2048});

            // 3. position_ids: [batch_size, sequence_length] 通常与 input_ids 第二维相同
            profile->setDimensions("position_ids", OptProfileSelector::kMIN, Dims2{1, 1});
            profile->setDimensions("position_ids", OptProfileSelector::kOPT, Dims2{1, 512});
            profile->setDimensions("position_ids", OptProfileSelector::kMAX, Dims2{1, 2048});

            for (int i = 0; i < 28; ++i) {
                std::string key_name = "past_key_values." + std::to_string(i) + ".key";
                std::string val_name = "past_key_values." + std::to_string(i) + ".value";

                profile->setDimensions(key_name.c_str(), OptProfileSelector::kMIN, Dims4{1, 4, 0, 128});
                profile->setDimensions(key_name.c_str(), OptProfileSelector::kOPT, Dims4{1, 4, 0, 128});
                profile->setDimensions(key_name.c_str(), OptProfileSelector::kMAX, Dims4{1, 4, 2048, 128});

                profile->setDimensions(val_name.c_str(), OptProfileSelector::kMIN, Dims4{1, 4, 0, 128});
                profile->setDimensions(val_name.c_str(), OptProfileSelector::kOPT, Dims4{1, 4, 0, 128});
                profile->setDimensions(val_name.c_str(), OptProfileSelector::kMAX, Dims4{1, 4, 2048, 128});
            }
            ret = config->addOptimizationProfile(profile);
            if(ret!=0){
                dylog::DynamicLogger().error("TensorRTCoreFramework::HotBuildEngine, addOptimizationProfile failed");
                return -1;
            }
            dylog::DynamicLogger().info("TensorRTCoreFramework::HotBuildEngine, addOptimizationProfile success");
            dylog::DynamicLogger().info("Building serialized engine...");
            auto serializedEngine = std::unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*network, *config));
            if (!serializedEngine) {
                dylog::DynamicLogger().error("buildSerializedNetwork failed");
                return -1;
            }
            std::string engine_path = engine.env["engine_path"].get<std::string>();
            std::ofstream engineFile(engine_path, std::ios::binary);
            if (!engineFile) {
                dylog::DynamicLogger().error("Cannot open engine file: {}", engine_path.c_str());
                return -1;
            }
            engineFile.write(static_cast<const char*>(serializedEngine->data()), static_cast<int64_t>(serializedEngine->size()));
            engineFile.close();
            dylog::DynamicLogger().info("Engine saved to {}", engine_path);
            return 0;

        }catch(const std::exception& e){
            dylog::DynamicLogger().error("TensorRTCoreFramework::HotBuildEngine, setDimensions failed: {}", e.what());
            return -1;
        }


        return 0;

    };

    auto save=[&]()
    {

    };

    ret = build_to_builder();
    if(ret!=0){
        std::cerr<<"TensorRTCoreFramework::HotBuildEngine, build_to_builder failed"<<std::endl;
        return -1;
    }
    return ret;
}

int gdlz::trt::core::TensorRTCoreFramework::InitShapeInfo(TensorBuilderEngineInfo& info)
{
    info.shapes=std::make_unique<std::vector<TensorRtBuilderShapeInfo>>();
    return 0;
}

int gdlz::trt::core::TensorRTCoreFramework::RegisterShapeInfo(const TensorRtBuilderShapeInfo& shape,TensorBuilderEngineInfo& info)
{
    info.shapes->push_back(shape);
    return 0;
}

int gdlz::trt::core::TensorRTCoreFramework::GetShapeInfoDetail(TensorBuilderEngineInfo& info)
{
    if(info.shapes==nullptr){
        dylog::DynamicLogger().error("TensorRTCoreFramework::GetShapeInfoDetail, shapes is nullptr");
        return -1;
    }
    for (auto shape : *info.shapes){

        std::string K_min_shape_str="[";
        for (int i=0;i<shape.K_min_dims;i++){
            K_min_shape_str+=" "+std::to_string(shape.K_min_shape[i]);
        }
        K_min_shape_str+=" ]";

        std::string K_max_shape_str="[";
        for (int i=0;i<shape.K_max_dims;i++) {
            K_max_shape_str+=" "+std::to_string(shape.K_max_shape[i]);
        }
        K_max_shape_str+=" ]";

        std::string K_opt_shape_str="[";
        for (int i=0;i<shape.K_opt_dims;i++) {
            K_opt_shape_str+=" "+std::to_string(shape.K_opt_shape[i]);
        }
        K_opt_shape_str+=" ]";


        dylog::DynamicLogger().info("shape: {}, K_min_shape: {}, K_max_shape: {}, K_opt_shape: {}",
            shape.name,
            K_min_shape_str.c_str(),
            K_max_shape_str.c_str(),
            K_opt_shape_str.c_str()
        );

    }
    return 0;
}

int gdlz::trt::core::TensorRTCoreFramework::BuildEngine(TensorRTCoreEngine& engine, const char* config_path,
                                                        TensorBuilderEngineInfo& info)
{
    auto dylog=dylog::DynamicLogger();
    using namespace nvinfer1;
    using namespace nvonnxparser;


    int ret = TensorRTCoreDirector::Handle(engine, config_path);
    if (ret != 0) {
        dylog.error("TensorRTCoreFramework::HotBuildEngine, Handle failed");
        return ret;
    }

    std::string module_path=engine.env["model_path"].get<std::string>();
    if(!engine.env.contains("engine_path")){
        dylog.error("TensorRTCoreFramework::HotBuildEngine, env: engine_path is missing");
        return -1;
    }
    dylog.debug("engine_path: {}", engine.env["engine_path"].get<std::string>().c_str());

    auto builder = std::unique_ptr<IBuilder>(createInferBuilder(engine.logger));
    if (!builder) {
        dylog.error("createInferBuilder failed");
        return -1;
    }

    auto flags = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(flags));
    if (!network) {
        dylog.error("TensorRTCoreFramework::HotBuildEngine, createNetworkV2 failed");
        return -1;
    }


    auto parser = std::unique_ptr<IParser>(createParser(*network, engine.logger));
    if (!parser) {
        dylog.error("TensorRTCoreFramework::HotBuildEngine, createParser failed");
        return -1;
    }
    if (!parser->parseFromFile(module_path.c_str(),static_cast<int>(ILogger::Severity::kWARNING))){
        dylog.error("TensorRTCoreFramework::HotBuildEngine, parseFromFile failed");
        return -1;
    }

    auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        dylog.error("TensorRTCoreFramework::HotBuildEngine, createBuilderConfig failed");
        return -1;
    }

    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 3ULL << 30);
    config->setFlag(BuilderFlag::kFP16);
    auto profile = builder->createOptimizationProfile();
    if (!profile) {
        dylog.error("TensorRTCoreFramework::HotBuildEngine, createOptimizationProfile failed");
        return -1;
    }
    for (auto shape : *info.shapes){
        auto min_dims=Dims();min_dims.nbDims=static_cast<int>(shape.K_min_dims);
        auto opt_dims=Dims();opt_dims.nbDims=static_cast<int>(shape.K_opt_dims);
        auto max_dims=Dims();max_dims.nbDims=static_cast<int>(shape.K_max_dims);

        for (int i=0;i<shape.K_min_dims;i++){min_dims.d[i]=shape.K_min_shape[i];}
        for (int i=0;i<shape.K_opt_dims;i++){opt_dims.d[i]=shape.K_opt_shape[i];}
        for (int i=0;i<shape.K_max_dims;i++){max_dims.d[i]=shape.K_max_shape[i];}

        profile->setDimensions(shape.name, OptProfileSelector::kMIN, min_dims);
        profile->setDimensions(shape.name, OptProfileSelector::kOPT, opt_dims);
        profile->setDimensions(shape.name, OptProfileSelector::kMAX, max_dims);
    }
    ret = config->addOptimizationProfile(profile);
    if(ret!=0){
        dylog.error("TensorRTCoreFramework::HotBuildEngine, addOptimizationProfile failed");
        return -1;
    }
    auto serializedEngine = std::unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!serializedEngine) {
        dylog.error("buildSerializedNetwork failed");
        return -1;
    }
    return 0;
}

int gdlz::trt::core::TensorRTCoreFramework::GetEngineInfo(TensorRTCoreEngine& engine)
{
    if (engine.engine == nullptr) {
        dylog::DynamicLogger().error("TensorRTCoreFramework::GetEngineInfo, engine is nullptr");
        return -1;
    }
    int nbBindings = engine.engine->getNbIOTensors();
    dylog::DynamicLogger().info("TensorRTCoreFramework::GetEngineInfo, nbBindings: {}", nbBindings);
    for (int i = 0; i < nbBindings; ++i)
    {
        const char* tensorName = engine.engine->getIOTensorName(i);
        auto type = engine.engine->getTensorDataType(tensorName);
        auto ioMode = engine.engine->getTensorIOMode(tensorName);
        auto dims = engine.engine->getTensorShape(tensorName);
        std::string shapeStr;
        for (int d = 0; d < dims.nbDims; ++d) {
            shapeStr += std::to_string(dims.d[d]);
            if (d != dims.nbDims - 1) shapeStr += ", ";
        }
        std::string typeStr;
        switch (type) {
            case nvinfer1::DataType::kINT32:
                typeStr = "INT32";
                break;
            case nvinfer1::DataType::kINT64:
                typeStr = "INT64";
                break;
            case nvinfer1::DataType::kFLOAT:
                typeStr = "FLOAT";
                break;
            case nvinfer1::DataType::kBF16:
                typeStr = "BF16";
                break;
            case nvinfer1::DataType::kHALF:
                typeStr = "HALF";
                break;
            default:
                typeStr = "UNKNOWN";
                break;
        }
        std::cout<<"tensorName: "<<tensorName<<", dims: "<<shapeStr<<", type: "<<typeStr<<std::endl;
    }

    return 0;
}

int gdlz::trt::core::TensorRTCoreFramework::CreateCtx(TensorRTCoreEngine& engine,TensorRTCoreCtx& ctx)
{
    if (engine.engine == nullptr) return code::operation::RT_FAILURE;
    try{
        ctx.context.reset();
        ctx.context.reset(engine.engine->createExecutionContext());
        if (ctx.context==nullptr) return code::ctx::CONTEXT_NULL;

    }
    catch (std::exception& e){
        std::cerr<<e.what()<<std::endl;
        return code::operation::FAILURE;
    }
    return code::operation::SUCCESS;
}


int gdlz::trt::core::TensorRTCoreFramework::Forward(TensorRTCoreCtx& ctx)
{
    if (ctx.context==nullptr) return code::ctx::CONTEXT_NULL;
    if (ctx.stream==nullptr) return -1;
    ctx.context->enqueueV3(*ctx.stream);
    return code::operation::SUCCESS;
}



int gdlz::trt::core::TensorRTCoreFramework::TestRun(TensorRTCoreEngine& engine)
{
    return 0;
}
