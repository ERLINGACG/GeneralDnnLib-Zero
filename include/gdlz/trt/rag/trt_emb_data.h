//
// Created by HP on 2026/4/19.
//

#ifndef GENERALDNNLIB_ZERO_TRT_EMB_DATA_H
#define GENERALDNNLIB_ZERO_TRT_EMB_DATA_H
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "NvInfer.h"
namespace gdlz::trt::rag {



    struct TrtEmbEngine
    {
        std::unique_ptr<nvinfer1::ICudaEngine> engine;
        std::unique_ptr<nvinfer1::IRuntime> runtime;
        std::vector<std::pair<std::string, std::string>> engine_info;
    };

    struct TrtEmbCtx {

        std::unique_ptr<nvinfer1::IExecutionContext> context;
        cudaStream_t* stream;
        std::unique_ptr<std::map<std::string, void*>> bindings;
        void reset(){

            if (stream != nullptr) {
                cudaStreamDestroy(*stream);
                delete stream;
                stream = nullptr;
            }

            if (context!=nullptr){
                context.reset();
            }

            if (bindings != nullptr) {
                bindings->clear();
            }
        }
    };

    struct  TrtEmbData {
        std::unique_ptr<float[]> embedding;
        int embedding_size;
    };
}
#endif //GENERALDNNLIB_ZERO_TRT_EMB_DATA_H
