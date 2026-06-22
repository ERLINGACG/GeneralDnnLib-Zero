//
// Created by HP on 2026/6/5.
//

#ifndef GENERALDNNLIB_ZERO_TRT_ILLM_FRAMEWORK_H
#define GENERALDNNLIB_ZERO_TRT_ILLM_FRAMEWORK_H
#include "trt_illm_config.h"
#include "trt_illm_data.h"
#include "gdlz/export.h"
#include "gdlz/trt/core/cuda/trt_cuda_data.h"

namespace gdlz::trt::illm{

    class GDLZ_CPP_API TensorRTIllmFramework{
           TensorRTIllmEngine  engine{};

        public:
            TensorRTIllmFramework(const char* config_path);
            ~TensorRTIllmFramework()=default;

            int CreateEngine();
            int GetEngineInfo();
            int RegisterMapping(const char* past_kv,const char* present_kv);
            int RegisterInput(const char* name,const char* type);
            int CreateTensorRTIllmCtx(TensorRTIllmCtx& ctx);

            static int SetSample(TensorRTIllmCtx& ctx,sample::TensorRTIllmSampleParams& params);


            static int Forward(TensorRTIllmCtx& ctx);

            static int ClearKvCache(TensorRTIllmCtx& ctx);

            static int GetKvCacheInfo(const TensorRTIllmCtx& ctx);

            static int AutoInputIds(
                TensorRTIllmCtx& ctx,
                void* input_ids,
                int64_t len
            );
            static int AutoPastKvCache(TensorRTIllmCtx& ctx,int64_t* d,int64_t d_len);
            static int AutoPresentKvCache(TensorRTIllmCtx& ctx);
            static int AutoLogits(TensorRTIllmCtx& ctx);
            static int PrefillDefault(TensorRTIllmCtx& ctx);
            static int DecodeDefault(TensorRTIllmCtx& ctx);
    };
}
#endif //GENERALDNNLIB_ZERO_TRT_ILLM_FRAMEWORK_H