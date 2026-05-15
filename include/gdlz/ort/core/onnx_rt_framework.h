//
// Created by HP on 2026/5/3.
//

#ifndef GENERALDNNLIB_ZERO_ONNX_RT_FRAMEWORK_H
#define GENERALDNNLIB_ZERO_ONNX_RT_FRAMEWORK_H
#include "gdlz/ort/data/onnx_rt_data.h"

namespace gdlz::ort::core{

    class OnnxRtFramework{

        data::OnnxRTEngineInfo engine_info;
      public:
        OnnxRtFramework() = default;
        ~OnnxRtFramework() = default;

        OnnxRtFramework* ResourceDirector(const char* config_path);

        int     CreateEngine(data::OnnxRTEngine& engine) const;
        static void    GetEngineInfo(data::OnnxRTEngine& ctx);
        void   run(data::OnnxRTEngine& ctx) const;
        void   run_batch_init(void* tokens,int len,data::OnnxRTEngine& ctx,data::OnnxExamKV& kv) const;
        void   run_token_one_step(data::OnnxRTEngine& ctx,data::OnnxExamKV& kv)  const;


        data::OnnxRTEngineInfo& GetEngineInfo_() {
            return engine_info;
        }

    };
}
#endif //GENERALDNNLIB_ZERO_ONNX_RT_FRAMEWORK_H