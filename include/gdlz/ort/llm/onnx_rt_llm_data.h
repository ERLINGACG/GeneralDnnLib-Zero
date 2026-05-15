//
// Created by HP on 2026/5/9.
//

#ifndef GENERALDNNLIB_ZERO_ONNX_RT_LLM_DATA_H
#define GENERALDNNLIB_ZERO_ONNX_RT_LLM_DATA_H
#include <vector>
#include <onnxruntime_cxx_api.h>

#include "gdlz/ort/data/onnx_rt_data.h"


namespace gdlz::ort::llm::data {

    struct OnnxRtLLmParam{
         int32_t top_k;
         float top_p;
         float temperature;
    };

    struct OnnxRtLLmEngine {
        std::shared_ptr<ort::data::OnnxRTEngine> core_engine;
        int32_t head_dim;
        int32_t heads;
        int32_t layers;
    };


    struct OnnxRtLLmCtx {
         std::unique_ptr<std::vector<std::string>>  input_names_ptr;
         std::unique_ptr<std::vector<std::string>>  output_names_ptr;
         std::unique_ptr<std::vector<int64_t>>      memory_ptr;
         int64_t token_pos;
         int64_t next_token_id;
         int32_t batch;

         ~OnnxRtLLmCtx()=default;
    };


    struct OnnxRtLLmKv {
        std::unique_ptr<std::vector<Ort::Value>> kv;
    };
}
#endif //GENERALDNNLIB_ZERO_ONNX_RT_LLM_DATA_H