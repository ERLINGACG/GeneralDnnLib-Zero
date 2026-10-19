//
// Created by HP on 2026/7/8.
//
#include "gdlz/trt/cv/core/trt_cv_core.h"

#include <iostream>

#include "gdlz/trt/core/trt_core_framework.h"
#include "gdlz/trt/core/cuda/op/trt_cuda_get.h"
#include "gdlz/trt/core/cuda/op/trt_cuda_memory.h"
#include "gdlz/trt/core/cuda/op/trt_cuda_post.h"
#include "gdlz/trt/cv/core/trt_cv_core_director.h"
#include "gdlz/trt/cv/core/private/trt_cv_core_impi.h"
using gdlz::trt::icv::core::TensorRTCvCoreFramework;
using gdlz::trt::icv::core::TensorRTCvCoreDirector;
using gdlz::trt::core::TensorRTCoreFramework;
using gdlz::trt::core::TensorRTCoreCtx;
using gdlz::trt::core::cuda::TensorRTCudaMemory;
using gdlz::trt::core::cuda::TensorRTCudaPost;
using gdlz::trt::core::cuda::TensorRTCudaGet;

using gdlz::trt::icv::core::TensorRTCvImpi;
using gdlz::trt::icv::core::TensorRTCvEngineImpi;


TensorRTCvCoreFramework::TensorRTCvCoreFramework(const char* config_path){
    // this->engine.impi();
    TensorRTCvCoreDirector::Handle(this->engine, config_path);
}

int TensorRTCvCoreFramework::CreateEngine(){
    return TensorRTCoreFramework::CreateEngine(*this->engine.impi());
}

int TensorRTCvCoreFramework::CreateCtx(TensorRTCvCtx& ctx) {
    if (ctx.impi() == nullptr || !ctx.impi()) {
        ctx.impi() = std::make_unique<TensorRTCvImpi>();
    }
    ctx.impi()->core_ctx = std::make_unique<TensorRTCoreCtx>();

    ctx.impi()->core_ctx->stream=new cudaStream_t;
    cudaStreamCreate(ctx.impi()->core_ctx->stream);
    if (!ctx.impi()->layer_names){
        ctx.impi()->layer_names=std::make_unique<std::map<std::string,std::string>>();
    }
    if (!ctx.impi()->layer_address){
        ctx.impi()->layer_address=std::make_unique<std::map<std::string,void*>>();
    }
    return  TensorRTCoreFramework::CreateCtx(*this->engine.impi(), *ctx.impi());
}

int TensorRTCvCoreFramework::GetEngineInfo(){
    auto& engine=this->engine.impi();
    return TensorRTCoreFramework::GetEngineInfo(engine->engine);
}

int TensorRTCvCoreFramework::RegisterInputLayer(TensorRTCvCtx& ctx, const char* name, const int64_t* shape,
                                                int64_t shape_len)
{
    ctx.impi()->layer_names->insert({name,"input"});
    return TensorRTCoreFramework::SetLayerShape(ctx.impi()->getCoreCtx(),name,shape,shape_len);
}

int TensorRTCvCoreFramework::ResetInputLayer(TensorRTCvCtx& ctx, const char* name, const int64_t* shape,
                                             int64_t shape_len)
{
    return TensorRTCoreFramework::SetLayerShape(ctx.impi()->getCoreCtx(), name, shape, shape_len);
}

int TensorRTCvCoreFramework::RegisterOutputLayer(TensorRTCvCtx& ctx, const char* name,
                                                 trt::core::cuda::DataType dataType) {
    ctx.impi()->layer_names->insert({name,"output"});

    auto dims=ctx.impi()->core_ctx->context->getTensorShape(name);
    int64_t size=1;
    for (int64_t i=0; i<dims.nbDims; i++){
        size *= dims.d[i];
    }
    void* address=nullptr;
    TensorRTCudaMemory::Malloc(size,address,dataType);
    ctx.impi()->core_ctx->context->setTensorAddress(name, address);
    ctx.impi()->layer_address->insert({name,address});

    return 0;
}

int TensorRTCvCoreFramework::ResetOutputLayer(TensorRTCvCtx& ctx, const char* name, trt::core::cuda::DataType dataType)
{
    if (ctx.impi()->layer_address->contains(name)){
        FreeLayer(ctx, name);
    }
    auto dims=ctx.impi()->core_ctx->context->getTensorShape(name);
    int64_t size=1;
    for (int64_t i=0; i<dims.nbDims; i++){
        size *= dims.d[i];
    }
    void* address=nullptr;
    TensorRTCudaMemory::Malloc(size,address,dataType);
    ctx.impi()->core_ctx->context->setTensorAddress(name, address);
    return 0;
}

int TensorRTCvCoreFramework::FreeLayer(TensorRTCvCtx& ctx, const char* name)
{
    if (ctx.impi()->layer_address->contains(name)){
        ctx.impi()->core_ctx->context->setTensorAddress(name,nullptr);
        return TensorRTCudaMemory::Free(ctx.impi()->layer_address->at(name));
    }
    return 0;
}


int TensorRTCvCoreFramework::PostImg(const char* name, TensorRTCvCtx& ctx, void* img, int64_t len,
                                     trt::core::cuda::DataType dataType)
{
    if (ctx.impi()->layer_address->contains(name)){
        FreeLayer(ctx, name);
    }
    void* address=nullptr;
    TensorRTCudaMemory::Malloc(len,address,dataType);
    TensorRTCudaPost::PostCuda(img,len,address,*ctx.impi()->core_ctx->stream,dataType);

    ctx.impi()->core_ctx->context->setTensorAddress(name,address);
    (*ctx.impi()->layer_address)[name] = address;
    return 0;
}

int TensorRTCvCoreFramework::BindAddress(const char* name, TensorRTCvCtx& ctx, void* address) {
    if (ctx.impi()->layer_address->contains(name)){
        auto old_address=ctx.impi()->layer_address->at(name);
        TensorRTCudaMemory::Free(old_address);
        ctx.impi()->layer_address->erase(name);
    }
    ctx.impi()->core_ctx->context->setTensorAddress(name, address);
    ctx.impi()->layer_address->insert({name,address});
    return 0;
}

int TensorRTCvCoreFramework::Forward(TensorRTCvCtx& ctx){
    return TensorRTCoreFramework::Forward(*ctx.impi());
}

int TensorRTCvCoreFramework::GetOutput(const char* name, TensorRTCvCtx& ctx, trt::core::cuda::TensorRTCudaData& data,
    trt::core::cuda::DataType dataType)
{
    auto* address=ctx.impi()->core_ctx->context->getTensorAddress(name);
    auto dims = ctx.impi()->core_ctx->context->getTensorShape(name);
    int64_t size=1;
    // std::cout<<"output shape: ";
    for (int64_t i=0; i<dims.nbDims; i++){
        // std::cout<<dims.d[i]<<" ";
        size *= dims.d[i];
    }


    auto res=TensorRTCudaGet::GetCudaFLOAT32(data,address,size,ctx.impi()->core_ctx->stream);
    return res;
}

void* TensorRTCvCoreFramework::GetAddress(const char* name, TensorRTCvCtx& ctx) {

    if (!ctx.impi()->layer_address->contains(name)){
        return nullptr;
    }
    return ctx.impi()->layer_address->at(name);
}

int64_t* TensorRTCvCoreFramework::GetOutputShapes(const char* name, TensorRTCvCtx& ctx)
{
    const auto [nbDims, d] = ctx.impi()->core_ctx->context->getTensorShape(name);
    auto* shape=new int64_t[nbDims];
    for (int64_t i=0; i<nbDims; i++){
        shape[i]=d[i];
    }
    return shape;
}

int64_t TensorRTCvCoreFramework::GetOutputShape(const char* name, TensorRTCvCtx& ctx)
{
    const auto [nbDims, d] = ctx.impi()->core_ctx->context->getTensorShape(name);
    return nbDims;
}
