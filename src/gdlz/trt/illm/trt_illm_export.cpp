//
// Created by HP on 2026/6/8.
//
#include "gdlz/trt/illm/trt_illm_export.h"


gdlz::trt::illm::TensorRTIllmFramework* gdlz::trt::illm::CreateTensorRTIllmFramework(const char* config_path){
    return new TensorRTIllmFramework(config_path);
}

void gdlz::trt::illm::DestroyTensorRTIllmFramework(const TensorRTIllmFramework* framework) {
    delete framework;
}

int gdlz::trt::illm::CreateTensorRTIllmEngine(TensorRTIllmFramework* framework) {
    return framework->CreateEngine();
}

int gdlz::trt::illm::GetTensorRTIllmEngineInfo(TensorRTIllmFramework* framework)
{
    return framework->GetEngineInfo();
}

int gdlz::trt::illm::CreateTensorRTIllmCtx(TensorRTIllmFramework* framework, TensorRTIllmCtx& ctx)
{
    return framework->CreateTensorRTIllmCtx(ctx);
}

int gdlz::trt::illm::RegisterKvMapping(TensorRTIllmFramework* framework, const char* past_kv, const char* present_kv)
{
    return framework->RegisterMapping(past_kv, present_kv);
}

int gdlz::trt::illm::SetSampler(TensorRTIllmCtx& ctx, sample::TensorRTIllmSampleParams params) {
    return TensorRTIllmFramework::SetSample(ctx, params);
}







int gdlz::trt::illm::ClearKvCache(TensorRTIllmCtx& ctx)
{
    return TensorRTIllmFramework::ClearKvCache(ctx);
}


int gdlz::trt::illm::RegisterInput(TensorRTIllmFramework* framework, const char* name, const char* type){
    return framework->RegisterInput(name, type);
}

int gdlz::trt::illm::AutoInputIds(TensorRTIllmCtx& ctx, void* input_ids, int64_t len) {
    return TensorRTIllmFramework::AutoInputIds(ctx, input_ids, len);
}

int gdlz::trt::illm::AutoPastKvCacheDefault(TensorRTIllmCtx& ctx, int64_t* d, int64_t d_len){
    return TensorRTIllmFramework::AutoPastKvCache(ctx, d, d_len);
}

int gdlz::trt::illm::AutoPresentKvCacheDefault(TensorRTIllmCtx& ctx){
    return TensorRTIllmFramework::AutoPresentKvCache(ctx);
}

int gdlz::trt::illm::AutoLogits(TensorRTIllmCtx& ctx) {
    return TensorRTIllmFramework::AutoLogits(ctx);
}

int gdlz::trt::illm::PrefillDefault(TensorRTIllmCtx& ctx) {
    return TensorRTIllmFramework::PrefillDefault(ctx);
}

int gdlz::trt::illm::DecodeDefault(TensorRTIllmCtx& ctx){
    return TensorRTIllmFramework::DecodeDefault(ctx);
}
