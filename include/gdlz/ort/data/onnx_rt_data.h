//
// Created by HP on 2026/5/3.
//

#ifndef GENERALDNNLIB_ZERO_ONNX_RT_DATA_H
#define GENERALDNNLIB_ZERO_ONNX_RT_DATA_H
#include <onnxruntime_cxx_api.h>

namespace gdlz::ort::data{
    struct OnnxRTEngine {
        Ort::Env env;
        std::string model_path;
        bool use_cuda;

    };

    struct OnnxRTCtx {
        std::unique_ptr<Ort::SessionOptions> sessionOptions;
        std::unique_ptr<Ort::Session> session;
        
        std::unique_ptr<OrtCUDAProviderOptions> cudaProviderOptions;
    };


    struct OnnxExamKV {

        std::vector<Ort::Value> next_past_kv;
        int token_count;
        int64_t next_token_id;
    };
}
#endif //GENERALDNNLIB_ZERO_ONNX_RT_DATA_H
