//
// Created by HP on 2026/5/3.
//

#ifndef GENERALDNNLIB_ZERO_ONNX_RT_FRAMEWORK_H
#define GENERALDNNLIB_ZERO_ONNX_RT_FRAMEWORK_H
#include "gdlz/ort/data/onnx_rt_data.h"

namespace gdlz::ort::core{

    class OnnxRtFramework{

        data::OnnxRTEngine engine;
      public:
        OnnxRtFramework() = default;
        ~OnnxRtFramework() = default;

        OnnxRtFramework* ResourceDirector(const char* config_path);

        int     CreateCtx(data::OnnxRTCtx& ctx) const;
        static void    GetEngineInfo(data::OnnxRTCtx& ctx);
        void   run(data::OnnxRTCtx& ctx) const;
        int    run_token(void* tokens,int len,data::OnnxRTCtx& ctx,data::OnnxExamKV& kv)  const;
        void   run_batch_init(void* tokens,int len,data::OnnxRTCtx& ctx,data::OnnxExamKV& kv) const;
        void   run_token_one_step(data::OnnxRTCtx& ctx,data::OnnxExamKV& kv)  const;


    };
}
#endif //GENERALDNNLIB_ZERO_ONNX_RT_FRAMEWORK_H