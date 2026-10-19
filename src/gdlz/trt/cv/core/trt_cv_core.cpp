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


#include "gdlz/trt/core/cuda/buffer/trt_cuda_buffer_oper.h"
#include "gdlz/trt/core/cuda/buffer/private/trt_cuda_buffer_impl.h"

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
    if (!ctx.impi()->layer_buffer){
        using gdlz::trt::core::cuda::TensorRTCudaBuffer;
        ctx.impi()->layer_buffer=std::make_unique<std::map<std::string,TensorRTCudaBuffer>>();
    }
    return  TensorRTCoreFramework::CreateCtx(*this->engine.impi(), *ctx.impi());
}

int TensorRTCvCoreFramework::GetEngineInfo(){
    auto& engine=this->engine.impi();
    return TensorRTCoreFramework::GetEngineInfo(engine->engine);
}

// int TensorRTCvCoreFramework::RegisterInputLayer(TensorRTCvCtx& ctx, const char* name, const int64_t* shape,
//                                                 int64_t shape_len)
// {
//     ctx.impi()->layer_names->insert({name,"input"});
//     return TensorRTCoreFramework::SetLayerShape(ctx.impi()->getCoreCtx(),name,shape,shape_len);
// }

int TensorRTCvCoreFramework::ResetInputLayer(TensorRTCvCtx& ctx, const char* name, const int64_t* shape,
                                             int64_t shape_len)
{
    return TensorRTCoreFramework::SetLayerShape(ctx.impi()->getCoreCtx(), name, shape, shape_len);
}

int TensorRTCvCoreFramework::ResetInputLayer(TensorRTCvCtx& ctx, const char* name, const int64_t* shape,
    int64_t shape_len, trt::core::cuda::BufferType bufferType)
{
    ResetInputLayer(ctx,name,shape,shape_len);
    const auto it = ctx.impi()->layer_buffer->find(name);
    if (it !=ctx.impi()->layer_buffer->end()){
        if (const auto address=resetSize(it->second,std::vector(shape,shape+shape_len),bufferType)){
            ctx.impi()->core_ctx->context->setTensorAddress(name, address);
        }
    }else {
        trt::core::cuda::TensorRTCudaBuffer buffer;
        if (const auto address=resetSize(buffer,std::vector(shape,shape+shape_len),bufferType)) {

            buffer.buffer->setStream(ctx.impi()->core_ctx->stream);
            ctx.impi()->core_ctx->context->setTensorAddress(name, address);
            ctx.impi()->layer_buffer->insert(
            {
                name,
                   std::move(buffer)
                }
            );
        }
    }
    return 0;
}

// int TensorRTCvCoreFramework::RegisterOutputLayer(TensorRTCvCtx& ctx, const char* name,
//                                                  trt::core::cuda::DataType dataType) {
//     ctx.impi()->layer_names->insert({name,"output"});
//
//     auto dims=ctx.impi()->core_ctx->context->getTensorShape(name);
//     int64_t size=1;
//     for (int64_t i=0; i<dims.nbDims; i++){
//         size *= dims.d[i];
//     }
//     void* address=nullptr;
//     TensorRTCudaMemory::Malloc(size,address,dataType);
//     ctx.impi()->core_ctx->context->setTensorAddress(name, address);
//     ctx.impi()->layer_address->insert({name,address});
//
//     return 0;
// }

// int TensorRTCvCoreFramework::ResetOutputLayer(TensorRTCvCtx& ctx, const char* name, trt::core::cuda::DataType dataType)
// {
//
//     const auto dims=ctx.impi()->core_ctx->context->getTensorShape(name);
//     std::vector<int64_t> shape;
//     for (int64_t i=0; i<dims.nbDims; i++){
//         shape.push_back(dims.d[i]);
//     }
//
//     using trt::core::cuda::resetSize;
//     using trt::core::cuda::TensorRTCudaBuffer;
//
//     const auto it = ctx.impi()->layer_buffer->find(name);
//     if (it !=ctx.impi()->layer_buffer->end()){
//         if (auto address=resetSize(it->second,shape,dataType)){
//             ctx.impi()->core_ctx->context->setTensorAddress(name, address);
//         }
//
//     }else {
//         TensorRTCudaBuffer buffer;
//
//         if (const auto address=resetSize(buffer,shape,dataType)) {
//
//             buffer.buffer->setStream(ctx.impi()->core_ctx->stream);
//             ctx.impi()->core_ctx->context->setTensorAddress(name, address);
//             ctx.impi()->layer_buffer->insert(
//             {
//                 name,
//                    std::move(buffer)
//                 }
//             );
//         }
//     }
//     return 0;
// }

int TensorRTCvCoreFramework::ResetOutputLayer(TensorRTCvCtx& ctx, const char* name, trt::core::cuda::BufferType bufferType)
{
    const auto dims=ctx.impi()->core_ctx->context->getTensorShape(name);
    std::vector<int64_t> shape;
    for (int64_t i=0; i<dims.nbDims; i++){
        shape.push_back(dims.d[i]);
    }

    using trt::core::cuda::resetSize;
    using trt::core::cuda::TensorRTCudaBuffer;

    const auto it = ctx.impi()->layer_buffer->find(name);
    if (it !=ctx.impi()->layer_buffer->end()){
        if (auto address=resetSize(it->second,shape,bufferType)){
            ctx.impi()->core_ctx->context->setTensorAddress(name, address);
        }

    }else {
        TensorRTCudaBuffer buffer;

        if (const auto address=resetSize(buffer,shape,bufferType)){
            buffer.buffer->setStream(ctx.impi()->core_ctx->stream);
            ctx.impi()->core_ctx->context->setTensorAddress(name, address);
            ctx.impi()->layer_buffer->insert(
            {
                name,
                   std::move(buffer)
                }
            );
        }
    }
    return 0;
}


void TensorRTCvCoreFramework::UploadData(const char* name, TensorRTCvCtx& ctx, const std::vector<std::byte>& host)
{
    using trt::core::cuda::uploadData;
    using trt::core::cuda::TensorRTCudaBuffer;

    const auto it = ctx.impi()->layer_buffer->find(name);
    if (it !=ctx.impi()->layer_buffer->end()){

        if (void* address=uploadData(it->second,host)){

            ctx.impi()->core_ctx->context->setTensorAddress(name,address);

         }else{

            std::cerr << "UploadData failed, name: " << name << std::endl;
            ctx.impi()->layer_buffer->erase(name);

         }

    }else {
        TensorRTCudaBuffer buffer;
        if (void* address=uploadData(buffer,host)){
            buffer.buffer->setStream(ctx.impi()->core_ctx->stream);
            ctx.impi()->core_ctx->context->setTensorAddress(name,address);
            ctx.impi()->layer_buffer->insert(
            {
                name,
                  std::move(buffer)
                }
            );
        }else{
            std::cerr << "UploadData failed, name: " << name << std::endl;
            ctx.impi()->layer_buffer->erase(name);
        }
    }

}

void TensorRTCvCoreFramework::DownloadData(const char* name, TensorRTCvCtx& ctx)
{
    using trt::core::cuda::downloadData;
    using trt::core::cuda::TensorRTCudaBuffer;
    const auto it = ctx.impi()->layer_buffer->find(name);
    if (it !=ctx.impi()->layer_buffer->end()){
        downloadData(it->second);
    }
}


int TensorRTCvCoreFramework::BindAddress(const char* name, TensorRTCvCtx& ctx, void* address) {
    ctx.impi()->core_ctx->context->setTensorAddress(name, address);
    return 0;
}

void const* TensorRTCvCoreFramework::GetLayerAddress(const char* name, TensorRTCvCtx& ctx) {
    return ctx.impi()->core_ctx->context->getTensorAddress(name);
}

int TensorRTCvCoreFramework::Forward(TensorRTCvCtx& ctx){
    return TensorRTCoreFramework::Forward(*ctx.impi());
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

gdlz::trt::core::cuda::TensorRTCudaBuffer& TensorRTCvCoreFramework::GetBuffer(const char* name, TensorRTCvCtx& ctx) {
    return ctx.impi()->layer_buffer->at(name);
}
