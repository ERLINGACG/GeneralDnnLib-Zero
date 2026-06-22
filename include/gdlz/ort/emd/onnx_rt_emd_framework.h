//
// Created by HP on 2026/5/22.
//

#ifndef GENERALDNNLIB_ZERO_ONNX_RT_EMD_FRAMEWORK_H
#define GENERALDNNLIB_ZERO_ONNX_RT_EMD_FRAMEWORK_H
#include "onnx_rt_emd_data.h"
#include "gdlz/export.h"
#include "gdlz/ort/core/onnx_rt_code_shape.h"
#include "gdlz/ort/data/onnx_rt_data.h"

namespace gdlz::ort::emd{
    class GDLZ_CPP_API  OnnxRtEmdFramework{

        data::OnnxRtEmdEngine engine{};
        ort::data::OnnxRTEngineInfo engine_info = {};
        private:
            int ResetEngine();
        public:
            OnnxRtEmdFramework(const char* config_path);
            ~OnnxRtEmdFramework();
            static int SetShape(core::shape::OnnxRtShape& shape,data::OnnxRtEmdCtx& ctx);
            static int GetAvgPoolEmd(data::OnnxRtEmdData& data,std::vector<Ort::Value>& emd);
            int GetEngineInfo();
            int Forward(data::OnnxRtEmdCtx& ctx, data::OnnxRtEmdData& data) const;


    };
}
#endif //GENERALDNNLIB_ZERO_ONNX_RT_EMD_FRAMEWORK_H
