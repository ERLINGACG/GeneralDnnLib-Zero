//
// Created by HP on 2026/7/2.
//
#include "gdlz/trt/cv/ocr/trt_cv_ocr_framework.h"

#include "gdlz/trt/core/trt_core_framework.h"
#include "gdlz/trt/core/cuda/op/trt_cuda_get.h"
#include "gdlz/trt/core/cuda/op/trt_cuda_memory.h"
#include "gdlz/trt/core/cuda/op/trt_cuda_post.h"
#include "gdlz/trt/cv/ocr/trt_cv_ocr_director.h"
using gdlz::trt::icv::ocr::TensorRTCvOCRFramework;

TensorRTCvOCRFramework::TensorRTCvOCRFramework(const char* config_path) {
    TensorRTCvOCRDirector::Handle(engine,config_path);
}

int TensorRTCvOCRFramework::CreateEngine() {
    return core::TensorRTCoreFramework::CreateEngine(this->engine);
}

int TensorRTCvOCRFramework::GetEngineInfo() {
    return core::TensorRTCoreFramework::GetEngineInfo(this->engine.engine);
}

int TensorRTCvOCRFramework::CreateCtx(TensorRTCvOCRCtx& ctx) {
    if (ctx.core_ctx==nullptr) {
        ctx.core_ctx=std::make_unique<core::TensorRTCoreCtx>();
    }
    ctx.core_ctx->stream=new cudaStream_t;
    cudaStreamCreate(ctx.core_ctx->stream);
    return core::TensorRTCoreFramework::CreateCtx<TensorRTCvOCREngine,TensorRTCvOCRCtx>(this->engine,ctx);
}

int TensorRTCvOCRFramework::RegisterInputLager(TensorRTCvOCRCtx& ctx, const char* name, const int64_t* shape,
                                               int64_t shape_len) {
    return core::TensorRTCoreFramework::SetLayerShape(ctx.getCoreCtx(),name,shape,shape_len);
}

int TensorRTCvOCRFramework::RegisterOutputLager(TensorRTCvOCRCtx& ctx, const char* name,core::cuda::DataType dataType) {
    auto dims=ctx.core_ctx->context->getTensorShape(name);
    int64_t size=1;
    std::cout<<"dims="<<dims.nbDims<<std::endl;
    for (int64_t i=0; i<dims.nbDims; i++){
        size *= dims.d[i];
        std::cout<<"dims.d["<<i<<"]="<<dims.d[i]<<std::endl;
    }
    void* address=nullptr;
    core::cuda::TensorRTCudaMemory::Malloc(size,address,dataType);
    ctx.core_ctx->context->setTensorAddress(name, address);
    return 0;
}

int TensorRTCvOCRFramework::PostImg(const char* name,TensorRTCvOCRCtx& ctx, void* img, int64_t len,
                                    core::cuda::DataType dataType) {

    void* address=nullptr;
    core::cuda::TensorRTCudaPost::PostCuda(img,len,address,*ctx.core_ctx->stream,dataType);
    ctx.core_ctx->context->setTensorAddress(name,address);
    return 0;
}

int TensorRTCvOCRFramework::Forward(TensorRTCvOCRCtx& ctx) {
    return core::TensorRTCoreFramework::Forward(ctx);
}

int TensorRTCvOCRFramework::GetOutput(const char* name, TensorRTCvOCRCtx& ctx, core::cuda::TensorRTCudaData& data,
    core::cuda::DataType dataType) {
    auto* address=ctx.core_ctx->context->getTensorAddress(name);
    auto dims = ctx.core_ctx->context->getTensorShape(name);
    int64_t size=1;
    for (int64_t i=0; i<dims.nbDims; i++){
        size *= dims.d[i];
        std::cout<<"dims.d["<<i<<"]="<<dims.d[i]<<std::endl;
    }

    auto res=core::cuda::TensorRTCudaGet::GetCudaFLOAT32(data,address,size,ctx.core_ctx->stream);
    std::cout<<"res="<<res<<std::endl;
    return 0;
}

