//
// Created by HP on 2026/5/9.
//

#ifndef GENERALDNNLIB_ZERO_ONNX_RT_LLM_DATA_H
#define GENERALDNNLIB_ZERO_ONNX_RT_LLM_DATA_H
#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <ostream>
#include <nlohmann/json_fwd.hpp>
#include <opencv2/core/hal/interface.h>

#include "onnx_rt_llm_sampler.h"
#include "gdlz/ort/data/onnx_rt_data.h"



namespace gdlz::ort::llm::data {



    struct InputShape {

        std::string type;
        std::string data_type;
        std::string name;
        int32_t layer_pos{};
        int32_t num{};
        std::vector<int64_t> shape;
    };

    struct KvShape
    {
        std::string type;
        std::vector<int64_t> shape;
        int32_t nums;
    };

    struct OnnxRtLLmEngine {
        ort::data::OnnxRTEngine core_engine;
        int32_t head_dim;
        int32_t heads;
        int32_t layers;
        std::vector<InputShape> input_shapes;

        [[nodiscard]] const ort::data::OnnxRTEngine& get_core_engine() const {
            return core_engine;
        }
    };

    struct OnnxRtLlmInput {

        // std::unique_ptr<int64_t[]>   tokens;
        std::unique_ptr<ort::data::OnnxRtInput> input;

        [[nodiscard]] auto& get_input() const{
            return *input;
        }
    };



    struct OnnxRtLLmCtx {

         std::unique_ptr<std::vector<int64_t>>      memory_ptr;
         std::unique_ptr<sampler::OnnxRtLLmSampler> sampler;
         int64_t token_pos;
         int64_t next_token_id;
         int32_t batch;

         ~OnnxRtLLmCtx()=default;

         void info() const{
             std::cout<<"token_pos:"<<token_pos<<std::endl;
             std::cout<<"next_token_id:"<<next_token_id<<std::endl;
             std::cout<<"batch:"<<batch<<std::endl;
             std::cout<<"memory_ptr size:"<<memory_ptr->size()<<std::endl;
         }

         void reset() {
             memory_ptr.reset();
             sampler.reset();
             token_pos = 0;
             next_token_id = 0;
             batch = 1;
         }

          void crop_memory_ptr(const int64_t len) {
             if(memory_ptr->size()>len && token_pos-len >= 0){
                 memory_ptr->erase(memory_ptr->begin(), memory_ptr->begin() + len);
                 token_pos -= len;
             }

         };
    };


    struct OnnxRtLLmKv {
        // float* data_ptr;
        std::unique_ptr<std::vector<std::vector<uint8_t>>>    prefixKvBuffer;
        std::unique_ptr<std::vector<Ort::Value>> prefixKv;
        std::unique_ptr<std::vector<std::vector<uint8_t>>>    KvBuffer;
        std::unique_ptr<std::vector<Ort::Value>> kv;
        void reset(){
            kv.reset();
            KvBuffer.reset();
            prefixKvBuffer.reset();
            prefixKv.reset();
        }
    };

    struct OnnxRtPrefillKv {
        // std::unique_ptr<std::vector<Ort：：>> kv;
    };
}
#endif //GENERALDNNLIB_ZERO_ONNX_RT_LLM_DATA_H