//
// Created by HP on 2026/4/19.
//
#include "gdlz/trt/rag/trt_emb_export.h"

gdlz::trt::rag::TrtEmbFramework* gdlz::trt::rag::CreateTrtEmbFramework(const char* config_path)
{
    auto framework = new TrtEmbFramework();
    framework->load([]{
        return TrtEmbDirector();
    },
        config_path
    );
    return framework;
}

void gdlz::trt::rag::DestroyTrtEmbFramework(const TrtEmbFramework* framework)
{
    delete framework;
}

void gdlz::trt::rag::GetEngineInfo(const TrtEmbFramework* framework)
{
    framework->GetEngineInfo();
}

int gdlz::trt::rag::CreateCtx(const TrtEmbFramework* framework, TrtEmbCtx& ctx)
{
    return framework->CreateCtx(ctx);
}

int gdlz::trt::rag::SetDims(TrtEmbCtx& ctx, const char* name, const int* input, int len)
{
    return TrtEmbFramework::SetDims(ctx,name, input, len);
}

int gdlz::trt::rag::GetDims(const TrtEmbFramework* framework, TrtEmbCtx& ctx, const char* name)
{
    return framework->GetDims(ctx,name);
}

int gdlz::trt::rag::PostCuda(
    const TrtEmbFramework* framework,
    const TrtEmbCtx& ctx, const char* name,
    const void* input, int len, int isMask
){
    return framework->PostCuda(ctx,name, input, len,isMask);
}

int gdlz::trt::rag::Forward(const TrtEmbFramework* framework, TrtEmbCtx& ctx) {
    return framework->Forward(ctx);
}

int gdlz::trt::rag::ForwardAsync(const TrtEmbFramework* framework, TrtEmbCtx& ctx) {
    return framework->ForwardAsync(ctx);
}

int gdlz::trt::rag::Synchronize(const TrtEmbCtx& ctx) {
    return TrtEmbFramework::Synchronize(ctx);
}

int gdlz::trt::rag::GetEmbedding(
    const TrtEmbFramework* framework,
    const TrtEmbCtx& ctx, TrtEmbData& embedding, const char* name)
{
    return framework->GetEmbedding(ctx,embedding,name);
}

int gdlz::trt::rag::GetPooledEmbedding(
    const TrtEmbFramework* framework,
    const TrtEmbCtx& ctx,
    TrtEmbData& embedding,
    const char* hidden_state_name,
    const char* attention_mask_name
){
    return framework->GetPooledEmbedding(ctx,embedding,hidden_state_name,attention_mask_name);
}

int gdlz::trt::rag::ClearBindings(TrtEmbCtx& ctx)
{
    return TrtEmbFramework::ClearBindings(ctx);
}


