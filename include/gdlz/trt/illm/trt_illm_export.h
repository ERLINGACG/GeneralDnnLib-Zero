//
// Created by HP on 2026/6/8.
//

#ifndef GENERALDNNLIB_ZERO_TRT_ILLM_EXPORT_H
#define GENERALDNNLIB_ZERO_TRT_ILLM_EXPORT_H
#include "trt_illm_framework.h"
#include "gdlz/export.h"
namespace gdlz::trt::illm{
    GDLZ_CORE_API TensorRTIllmFramework* CreateTensorRTIllmFramework(const char* config_path);
    GDLZ_CORE_API void DestroyTensorRTIllmFramework(const TensorRTIllmFramework* framework);
    GDLZ_CORE_API int CreateTensorRTIllmEngine(TensorRTIllmFramework* framework);
    GDLZ_CORE_API int GetTensorRTIllmEngineInfo(TensorRTIllmFramework* framework);

    GDLZ_CORE_API int CreateTensorRTIllmCtx(TensorRTIllmFramework* framework,TensorRTIllmCtx& ctx);
    GDLZ_CORE_API int RegisterKvMapping(TensorRTIllmFramework* framework,const char* past_kv, const char* present_kv);
    GDLZ_CORE_API int SetSampler           (TensorRTIllmCtx& ctx, sample::TensorRTIllmSampleParams params);


    GDLZ_CORE_API int ClearKvCache  (TensorRTIllmCtx& ctx);


    GDLZ_CORE_API int RegisterInput(TensorRTIllmFramework* framework,const char* name, const char* type);
    GDLZ_CORE_API int AutoInputIds(TensorRTIllmCtx& ctx,void* input_ids,int64_t len);
    GDLZ_CORE_API int AutoPastKvCacheDefault(TensorRTIllmCtx& ctx,int64_t* d,int64_t d_len);
    GDLZ_CORE_API int AutoPresentKvCacheDefault(TensorRTIllmCtx& ctx);
    GDLZ_CORE_API int AutoLogits(TensorRTIllmCtx& ctx);
    GDLZ_CORE_API int PrefillDefault(TensorRTIllmCtx& ctx);
    GDLZ_CORE_API int DecodeDefault(TensorRTIllmCtx& ctx);

}
#endif //GENERALDNNLIB_ZERO_TRT_ILLM_EXPORT_H