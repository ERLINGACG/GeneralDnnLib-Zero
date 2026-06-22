//
// Created by HP on 2026/6/5.
//

#ifndef GENERALDNNLIB_ZERO_TRT_CORE_DATA_H
#define GENERALDNNLIB_ZERO_TRT_CORE_DATA_H
#include <memory>
#include <NvInferRuntime.h>
#include <string>
#include <vector>
#include <dylog/logger.h>
#include <nlohmann/json.hpp>

#include "cuda/trt_cuda_data.h"

namespace gdlz::trt::core{

    class TrtLogger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override {
            if (severity == Severity::kINTERNAL_ERROR || severity == Severity::kERROR) {
                dylog::DynamicLogger().error("TensorRT: {}", msg);
            } else if (severity == Severity::kWARNING) {
                dylog::DynamicLogger().warn("TensorRT: {}", msg);
            } else if (severity == Severity::kINFO) {
                dylog::DynamicLogger().info("TensorRT: {}", msg);
            } else {
                dylog::DynamicLogger().debug("TensorRT: {}", msg);
            }
        }

    };


    struct TensorRTCoreEngine{
        std::unique_ptr<nvinfer1::ICudaEngine> engine{};
        std::unique_ptr<nvinfer1::IRuntime> runtime{};
        nlohmann::json config{};
        nlohmann::json env{};
        TrtLogger logger=TrtLogger();
        std::vector<std::pair<std::string, std::string>> engine_info{};

        auto& getCoreEngine(){
            return *this;
        }
    };

    struct ExportCoreEngine{
        std::unique_ptr<TensorRTCoreEngine> engine{};
        [[nodiscard]] auto& getCoreEngine(){
            return *engine;
        }
    };

    struct TensorRtShape{
        char*    name;
        int64_t  dims;
        int64_t* shape;

        bool buildDims(nvinfer1::Dims& dims64) const {
            if (dims<=0) return false;
            dims64.nbDims=static_cast<int32_t>(dims);
            for (int i = 0; i <dims; ++i) dims64.d[i] = shape[i];
            return true;
        }
    };


    struct TensorRtBuilderShapeInfo{
        char*    name;

        int64_t  K_min_dims;
        int64_t* K_min_shape;

        int64_t  K_max_dims;
        int64_t* K_max_shape;

        int64_t  K_opt_dims;
        int64_t* K_opt_shape;
    };

    struct TensorBuilderEngineInfo{
        std::unique_ptr<std::vector<TensorRtBuilderShapeInfo>> shapes;
    };

    struct TensorRTCoreCtx{
        std::unique_ptr<nvinfer1::IExecutionContext> context;
        cudaStream_t* stream;

        auto& getCoreCtx(){
            return *this;
        }
    };
}
#endif //GENERALDNNLIB_ZERO_TRT_CORE_DATA_H