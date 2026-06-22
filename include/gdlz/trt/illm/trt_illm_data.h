//
// Created by HP on 2026/6/5.
//

#ifndef GENERALDNNLIB_ZERO_TRT_ILLM_DATA_H
#define GENERALDNNLIB_ZERO_TRT_ILLM_DATA_H
#include "gdlz/trt/core/trt_core_data.h"
#include "sample/sp/trt_illm_sample.h"


namespace gdlz::trt::illm {
    struct TensorRTIllmEngine {
       core::TensorRTCoreEngine engine{};
       std::vector<std::pair<std::string,std::string>> kv_mapping{};
       std::vector<std::pair<std::string,std::string>> input_inputs{};

       auto& getCoreEngine(){
           return engine;
       }
    };

    // struct TensorRTIllmInput{
    //     std::string name;
    //     std::string type;
    // };

    struct TensorRTIllmCtx {

        std::unique_ptr<core::TensorRTCoreCtx>      core_ctx;
        std::unique_ptr<sample::TensorRTIllmSample> sample;
        std::unique_ptr<std::vector<std::pair<std::string,std::string>>> input_inputs;
        std::unique_ptr<std::vector<std::pair<std::string,std::string>>> kv_mapping;

        // int64_t seq_len;
        // int64_t heads;
        // int64_t head_dim;
        // int64_t batch;
        int64_t next_token;
        int64_t abs_token_pos;

        [[nodiscard]] auto& getCoreCtx() const{
            return *core_ctx;
        }
    };





}
#endif //GENERALDNNLIB_ZERO_TRT_ILLM_DATA_H