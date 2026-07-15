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
using gdlz::trt::icv::core::TensorRTCvCoreFramework;
using gdlz::trt::icv::core::TensorRTCvCoreDirector;
using gdlz::trt::core::TensorRTCoreFramework;
using gdlz::trt::core::TensorRTCoreCtx;
using gdlz::trt::core::cuda::TensorRTCudaMemory;
using gdlz::trt::core::cuda::TensorRTCudaPost;
using gdlz::trt::core::cuda::TensorRTCudaGet;

TensorRTCvCoreFramework::TensorRTCvCoreFramework(const char* config_path){
    TensorRTCvCoreDirector::Handle(this->engine, config_path);
}

int TensorRTCvCoreFramework::CreateEngine(){
    return TensorRTCoreFramework::CreateEngine(this->engine);
}

int TensorRTCvCoreFramework::CreateCtx(TensorRTCvCtx& ctx) {
    if (ctx.core_ctx==nullptr || !ctx.core_ctx) {
        ctx.core_ctx=std::make_unique<TensorRTCoreCtx>();
    }
    ctx.core_ctx->stream=new cudaStream_t;
    cudaStreamCreate(ctx.core_ctx->stream);
    if (!ctx.layer_names){
        ctx.layer_names=std::make_unique<std::map<std::string,std::string>>();
    }
    if (!ctx.layer_address){
        ctx.layer_address=std::make_unique<std::map<std::string,void*>>();
    }
    return  TensorRTCoreFramework::CreateCtx(this->engine,ctx);
}

int TensorRTCvCoreFramework::GetEngineInfo(){
    return TensorRTCoreFramework::GetEngineInfo(this->engine.engine);
}

int TensorRTCvCoreFramework::RegisterInputLayer(TensorRTCvCtx& ctx, const char* name, const int64_t* shape,
                                                int64_t shape_len)
{
    ctx.layer_names    ->insert({name,"input"});
    return TensorRTCoreFramework::SetLayerShape(ctx.getCoreCtx(),name,shape,shape_len);
}

int TensorRTCvCoreFramework::RegisterOutputLayer(TensorRTCvCtx& ctx, const char* name,
    trt::core::cuda::DataType dataType) {
    ctx.layer_names->insert({name,"output"});

    auto dims=ctx.core_ctx->context->getTensorShape(name);
    int64_t size=1;
    for (int64_t i=0; i<dims.nbDims; i++){
        size *= dims.d[i];
    }
    void* address=nullptr;
    TensorRTCudaMemory::Malloc(size,address,dataType);
    ctx.core_ctx->context->setTensorAddress(name, address);
    ctx.layer_address->insert({name,address});

    return 0;
}

int TensorRTCvCoreFramework::FreeLayer(TensorRTCvCtx& ctx, const char* name)
{
    if (ctx.layer_address->contains(name)){
        ctx.core_ctx->context->setTensorAddress(name,nullptr);
        return TensorRTCudaMemory::Free(ctx.layer_address->at(name));
    }
    return 0;
}


int TensorRTCvCoreFramework::PostImg(const char* name, TensorRTCvCtx& ctx, void* img, int64_t len,
                                     trt::core::cuda::DataType dataType)
{
    if (ctx.layer_address->contains(name)){
        FreeLayer(ctx, name);
    }
    void* address=nullptr;
    TensorRTCudaMemory::Malloc(len,address,dataType);
    TensorRTCudaPost::PostCuda(img,len,address,*ctx.core_ctx->stream,dataType);

    ctx.core_ctx->context->setTensorAddress(name,address);
    (*ctx.layer_address)[name] = address;
    return 0;
}

int TensorRTCvCoreFramework::Forward(TensorRTCvCtx& ctx){
    return TensorRTCoreFramework::Forward(ctx);
}

int TensorRTCvCoreFramework::GetOutput(const char* name, TensorRTCvCtx& ctx, trt::core::cuda::TensorRTCudaData& data,
    trt::core::cuda::DataType dataType)
{
    auto* address=ctx.core_ctx->context->getTensorAddress(name);
    auto dims = ctx.core_ctx->context->getTensorShape(name);
    int64_t size=1;
    for (int64_t i=0; i<dims.nbDims; i++){
        size *= dims.d[i];
        std::cout<<"dims.d["<<i<<"]="<<dims.d[i]<<std::endl;
    }

    auto res=TensorRTCudaGet::GetCudaFLOAT32(data,address,size,ctx.core_ctx->stream);
    std::cout<<"res="<<res<<std::endl;
    return 0;
}
