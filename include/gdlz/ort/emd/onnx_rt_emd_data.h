//
// Created by HP on 2026/5/22.
//

#ifndef GENERALDNNLIB_ZERO_ONNX_RT_EMD_DATA_H
#define GENERALDNNLIB_ZERO_ONNX_RT_EMD_DATA_H
#include "gdlz/ort/core/onnx_rt_code_shape.h"
#include "gdlz/ort/data/onnx_rt_data.h"

namespace gdlz::ort::emd::data {
    struct OnnxRtEmdEngine {
        ort::data::OnnxRTEngine core_engine;

        auto& get_core_engine(){
            return core_engine;
        }
        void Reset()
        {
            core_engine.session.reset();
            core_engine.cudaProviderOptions.reset();
            core_engine.sessionOptions.reset();
            core_engine.input_name_ptr.reset();
            core_engine.output_name_ptr.reset();
        }
    };

    struct OnnxRtEmdCtx{

        std::unique_ptr<std::vector<core::shape::OnnxRtShape>> shape;
    };

    struct OnnxRtEmdData{
        float* embedding;
        int    embedding_size;
    };

}
#endif //GENERALDNNLIB_ZERO_ONNX_RT_EMD_DATA_H