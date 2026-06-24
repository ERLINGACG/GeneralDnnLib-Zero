//
// Created by HP on 2026/6/5.
//
#include "gdlz/trt/illm/trt_illm_framework.h"

#include <iostream>

#include "gdlz/trt/core/trt_code.h"
#include "gdlz/trt/core/trt_core_framework.h"
#include "gdlz/trt/core/cuda/op/trt_cuda_get.h"
#include "gdlz/trt/core/cuda/op/trt_cuda_memory.h"
#include "gdlz/trt/core/cuda/op/trt_cuda_post.h"
#include "gdlz/trt/illm/trt_illm_director.h"
#include "gdlz/trt/illm/sample/cu/trt_illm_temp.cuh"


using gdlz::trt::illm::TensorRTIllmFramework;
using gdlz::trt::illm::TensorRTIllmDirector;
using gdlz::trt::core::TensorRTCoreFramework;

TensorRTIllmFramework::TensorRTIllmFramework(const char* config_path) {
      TensorRTIllmDirector::Handle(engine,config_path);
}

int TensorRTIllmFramework::CreateEngine() {
    return TensorRTCoreFramework::CreateEngine<TensorRTIllmEngine>(engine);
}

int TensorRTIllmFramework::CreateTensorRTIllmCtx(TensorRTIllmCtx& ctx) {
    if (ctx.core_ctx==nullptr) {
        ctx.core_ctx=std::make_unique<core::TensorRTCoreCtx>();
    }
    ctx.core_ctx->stream=new cudaStream_t;
    cudaStreamCreate(ctx.core_ctx->stream);
    ctx.kv_mapping=std::make_unique<std::vector<std::pair<std::string,std::string>>>(engine.kv_mapping);
    ctx.input_inputs=std::make_unique<std::vector<std::pair<std::string,std::string>>>(engine.input_inputs);
    ctx.next_token=0;
    ctx.abs_token_pos=0;
    return TensorRTCoreFramework::CreateCtx<TensorRTIllmEngine,TensorRTIllmCtx>(engine,ctx);
}

int TensorRTIllmFramework::SetSample(TensorRTIllmCtx& ctx,sample::TensorRTIllmSampleParams& params) {
    ctx.sample=std::make_unique<sample::TensorRTIllmSample>(params);
    if (ctx.sample==nullptr){
        return code::operation::FAILURE;
    }
    return code::operation::SUCCESS;
}

int TensorRTIllmFramework::GetEngineInfo(){
    std::cout<<"input_inputs :"<<std::endl;
    for (auto& item : engine.input_inputs) {
       std::cout<<item.first<<" "<<item.second<<std::endl;
   }
   for (auto& item : engine.kv_mapping) {
       std::cout<<item.first<<" "<<item.second<<std::endl;
   }

   return  TensorRTCoreFramework::GetEngineInfo(this->engine.engine);
}

int TensorRTIllmFramework::RegisterMapping(const char* past_kv, const char* present_kv){
    engine.kv_mapping.emplace_back(past_kv,present_kv);
    return code::operation::SUCCESS;
}

int TensorRTIllmFramework::RegisterInput(const char* name, const char* type) {
    engine.input_inputs.emplace_back(name,type);
    return code::operation::SUCCESS;
}

int TensorRTIllmFramework::RegisterLogits(const char* name, const char* type) {
    engine.input_inputs.emplace_back(name,type);
    return code::operation::SUCCESS;
}


int TensorRTIllmFramework::Forward(TensorRTIllmCtx& ctx) {
    TensorRTCoreFramework::Forward<TensorRTIllmCtx>(ctx);
    return code::operation::SUCCESS;
}



int TensorRTIllmFramework::ClearKvCache(TensorRTIllmCtx& ctx){
    return code::operation::SUCCESS;
}

int TensorRTIllmFramework::GetKvCacheInfo(const TensorRTIllmCtx& ctx){

    return 0;
}

int TensorRTIllmFramework::AutoInputIds(TensorRTIllmCtx& ctx, void* input_ids,
    int64_t len) {
    auto* input_ids_int64=static_cast<int64_t*>(input_ids);
    std::vector<int64_t> input_ids_vector(input_ids_int64,input_ids_int64+len);
    std::vector<int64_t> att_mask(len,1);
    std::vector<int64_t> pos(len);
    void* input_address=nullptr;
    void* attention_mask_address=nullptr;

    // ctx.seq_len=len;

    core::cuda::TensorRTCudaPost::PostCuda(
        input_ids_vector.data(),
        static_cast<int64_t>(input_ids_vector.size()),
        input_address,*ctx.core_ctx->stream,core::cuda::DataType::INT64
    );

    core::cuda::TensorRTCudaPost::PostCuda(
        att_mask.data(),static_cast<int64_t>(att_mask.size()),
        attention_mask_address,
        *ctx.core_ctx->stream,
        core::cuda::DataType::INT64
    );

    for (auto& [name,type] : *ctx.input_inputs){

        if (type=="INP"){
            std::cout<<"set input ids for"<<name<<std::endl;
            ctx.core_ctx->context->setInputShape(name.c_str(),
                nvinfer1::Dims{2,1,
                    static_cast<int64_t>(input_ids_vector.size())}
                );
            ctx.core_ctx->context->setTensorAddress(name.c_str(),input_address);
        }

        if (type=="MSK"){
            std::cout<<"set attention mask for"<<name<<std::endl;
            ctx.core_ctx->context->setInputShape(name.c_str(),
                nvinfer1::Dims{2,1,
                    static_cast<int64_t>(att_mask.size())}
                );
            ctx.core_ctx->context->setTensorAddress(name.c_str(),attention_mask_address);
        }

        if (type=="POS") {
            std::cout<<"set pos for"<<name<<std::endl;
            void* pos_address=nullptr;
            for (int i = 0; i < pos.size(); ++i){
                pos[i]=i;
            }

            core::cuda::TensorRTCudaPost::PostCuda(
                pos.data(),static_cast<int64_t>(pos.size()),
                pos_address,
                *ctx.core_ctx->stream,
                core::cuda::DataType::INT64
            );

            ctx.core_ctx->context->setTensorAddress(name.c_str(),pos_address);
            ctx.core_ctx->context->setInputShape(name.c_str(),
                nvinfer1::Dims{2,1,
                    static_cast<int64_t>(pos.size())
                }
            );

        }
    }

    return code::operation::SUCCESS;
}

int TensorRTIllmFramework::AutoPastKvCache(TensorRTIllmCtx& ctx, int64_t* d, int64_t d_len){
    auto dims=nvinfer1::Dims();
    dims.nbDims=static_cast<int32_t>(d_len);
    size_t size=1;
    for (int32_t i = 0; i < dims.nbDims; ++i){dims.d[i] = d[i];size*=dims.d[i];}

    for (auto& past_kv : *ctx.kv_mapping | std::views::keys){
        ctx.core_ctx->context->setInputShape(past_kv.c_str(),dims);
        if (size==0){
            void* kv_address=nullptr;
            std::vector<float> present_kv_vector(1,0.0f);
            core::cuda::TensorRTCudaPost::PostCuda(
                present_kv_vector.data(),1,
                kv_address,*ctx.core_ctx->stream,core::cuda::DataType::FLOAT32
            );
            ctx.core_ctx->context->setTensorAddress(past_kv.c_str(),kv_address);
        }
    }
    return code::operation::SUCCESS;
}

int TensorRTIllmFramework::AutoPastKvCache(TensorRTIllmCtx& ctx, const char* name, int64_t* d, int64_t d_len,size_t shape_type)
{
    TensorRTCoreFramework::SetLayerShape(ctx,name,d,d_len);
    void* kv_address=nullptr;
    std::vector<float> present_kv_vector(1,0.0f);
    core::cuda::TensorRTCudaPost::PostCuda(
        present_kv_vector.data(),1,
        kv_address,*ctx.core_ctx->stream,static_cast<core::cuda::DataType>(shape_type)
    );
    ctx.core_ctx->context->setTensorAddress(name,kv_address);
    return 0;
}

int TensorRTIllmFramework::AutoPresentKvCache(TensorRTIllmCtx& ctx)
{
    for (auto& present_kv : *ctx.kv_mapping | std::views::values){
        auto dims=ctx.core_ctx->context->getTensorShape(present_kv.c_str());
        int64_t size=1;
        std::cout<<"dims.nbDims="<<dims.nbDims<<std::endl;
        std::cout<<"dims.d[";
        for (int32_t i = 0; i < dims.nbDims; ++i){
            std::cout<<" "<<dims.d[i];
            size*=dims.d[i];
        }
        std::cout<<"]"<<std::endl;
        void* present_address=nullptr;
        core::cuda::TensorRTCudaMemory::Malloc(
            size, present_address, core::cuda::DataType::FLOAT32
        );
        ctx.core_ctx->context->setTensorAddress(present_kv.c_str(),present_address);
    }
    return code::operation::SUCCESS;
}

int TensorRTIllmFramework::AutoPresentKvCache(TensorRTIllmCtx& ctx, const char* name,size_t shape_type)
{
    auto dims=ctx.core_ctx->context->getTensorShape(name);
    int64_t size=1;
    for (int32_t i = 0; i < dims.nbDims; ++i){size*=dims.d[i];}
    void* kv_address=nullptr;
    core::cuda::TensorRTCudaMemory::Malloc(
        size, kv_address, static_cast<core::cuda::DataType>(shape_type)
    );
    ctx.core_ctx->context->setTensorAddress(name,kv_address);
    return code::operation::SUCCESS;
}

int TensorRTIllmFramework::AutoLogits(TensorRTIllmCtx& ctx)
{
    for (auto& [name,type] : *ctx.input_inputs){
        if (type=="LOG"){
           auto dims= ctx.core_ctx->context->getTensorShape(name.c_str());
            int64_t size=1;
            std::cout<<"dims.nbDims="<<dims.nbDims<<std::endl;
            std::cout<<"dims.d[";
            for (int32_t i = 0; i < dims.nbDims; ++i){
                std::cout<<" "<<dims.d[i];
                size*=dims.d[i];
            }
            std::cout<<"]"<<std::endl;
            void* logits_address=nullptr;
            core::cuda::TensorRTCudaMemory::Malloc(
                size, logits_address, core::cuda::DataType::FLOAT32
            );
            ctx.core_ctx->context->setTensorAddress(name.c_str(),logits_address);
        }
    }
    return code::operation::SUCCESS;
}

int TensorRTIllmFramework::PrefillDefault(TensorRTIllmCtx& ctx)
{
    Forward(ctx);
    core::cuda::TensorRTCudaMemory::Sync(ctx);
    for (auto& [name,type] : *ctx.input_inputs) {
        if (type=="LOG"){
            auto logits_address=ctx.core_ctx->context->getTensorAddress(name.c_str());
            if (logits_address!=nullptr){

                auto dims=ctx.core_ctx->context->getTensorShape(name.c_str());
                if (dims.nbDims!=3){
                    return code::shap::SHAPE_DIMS_FAILURE;
                }
                const auto v_s=dims.d[2];
                const auto token_pos=dims.d[1];
                const auto last_abs_token_offest=(token_pos-1)*v_s;
                ctx.next_token=ctx.sample->SampleTemperature(
                    logits_address,
                    last_abs_token_offest,
                    v_s
                );
                ctx.abs_token_pos+=dims.d[1];
                std::cout<<"ctx.next_token="<<ctx.next_token<<std::endl;
                std::cout<<"ctx.abs_token_pos="<<ctx.abs_token_pos<<std::endl;
            }
        }
    }
    return code::operation::SUCCESS;
}

int TensorRTIllmFramework::DecodeDefault(TensorRTIllmCtx& ctx) {

    std::vector<int64_t> input_ids_vector={ctx.next_token};
    std::vector<int64_t> mask_ids_vector ={1};
    std::vector<int64_t> pos             ={ctx.abs_token_pos};
    auto logits_name="";
    for (auto& [name,type] : *ctx.input_inputs) {
        if (type=="LOG"){
            logits_name=name.c_str();
        }
        if (type=="INP"){
            auto last_address=ctx.core_ctx->context->getTensorAddress(name.c_str());
            core::cuda::TensorRTCudaMemory::Free(last_address);
            ctx.core_ctx->context->setInputShape(name.c_str(),
                nvinfer1::Dims{2,1,static_cast<int64_t>(input_ids_vector.size())}
            );
            void* input_address=nullptr;
            core::cuda::TensorRTCudaPost::PostCuda(
                input_ids_vector.data(),1,
                input_address,*ctx.core_ctx->stream,core::cuda::DataType::INT64
            );
            ctx.core_ctx->context->setTensorAddress(name.c_str(),input_address);
        }

        if (type=="MSK") {
            auto last_address=ctx.core_ctx->context->getTensorAddress(name.c_str());
            core::cuda::TensorRTCudaMemory::Free(last_address);
            ctx.core_ctx->context->setInputShape(
                name.c_str(),
                nvinfer1::Dims{2,1,static_cast<int64_t>(mask_ids_vector.size())}
            );
            void* mask_address=nullptr;
            core::cuda::TensorRTCudaPost::PostCuda(
                mask_ids_vector.data(),1,
                mask_address,*ctx.core_ctx->stream,core::cuda::DataType::INT64
            );
            ctx.core_ctx->context->setTensorAddress(name.c_str(),mask_address);
        }

        if (type=="POS") {
            auto last_address=ctx.core_ctx->context->getTensorAddress(name.c_str());
            core::cuda::TensorRTCudaMemory::Free(last_address);
            ctx.core_ctx->context->setInputShape(
                name.c_str(),
                nvinfer1::Dims{2,1,static_cast<int64_t>(pos.size())}
            );
            void* pos_address=nullptr;
            core::cuda::TensorRTCudaPost::PostCuda(
                pos.data(),1,
                pos_address,*ctx.core_ctx->stream,core::cuda::DataType::INT64
            );
            ctx.core_ctx->context->setTensorAddress(name.c_str(),pos_address);
        }
    }
    for (auto& [past_kv,present_kv] : *ctx.kv_mapping ){

        auto last_kv_address=ctx.core_ctx->context->getTensorAddress(past_kv.c_str());
        core::cuda::TensorRTCudaMemory::Free(last_kv_address);

        auto past_kv_address=ctx.core_ctx->context->getTensorAddress(present_kv.c_str());
        ctx.core_ctx->context->setTensorAddress(past_kv.c_str(),const_cast<void*>(past_kv_address));
        ctx.core_ctx->context->setInputShape(past_kv.c_str(),
            nvinfer1::Dims{4,1,4,ctx.abs_token_pos,128}
        );
    }
    for (auto& present_kv : *ctx.kv_mapping | std::views::values){

        auto dims = ctx.core_ctx->context->getTensorShape(present_kv.c_str());
        void* present_address=nullptr;

        // std::cout<<"dims.d[";
        int64_t size=1;
        for (int32_t i = 0; i < dims.nbDims; ++i){
            // std::cout<<" "<<dims.d[i];
            size*=dims.d[i];
        }

        core::cuda::TensorRTCudaMemory::Malloc(size,present_address,core::cuda::DataType::FLOAT32);
        ctx.core_ctx->context->setTensorAddress(present_kv.c_str(),present_address);
    }

    if (logits_name!=""){
        auto log_dims=ctx.core_ctx->context->getTensorShape(logits_name);
        int64_t size=1;
        for (int32_t i = 0; i < log_dims.nbDims; ++i){
            size*=log_dims.d[i];
        }
        void* log_address=nullptr;
        core::cuda::TensorRTCudaMemory::Malloc(size,log_address,core::cuda::DataType::FLOAT32);
        ctx.core_ctx->context->setTensorAddress(logits_name,log_address);
    }

    Forward(ctx);
    core::cuda::TensorRTCudaMemory::Sync(ctx);
    auto logits_address=ctx.core_ctx->context->getTensorAddress(logits_name);

    if (logits_address!=nullptr){

        auto dims=ctx.core_ctx->context->getTensorShape(logits_name);
        if (dims.nbDims!=3){
            return code::shap::SHAPE_DIMS_FAILURE;
        }
        const auto v_s=dims.d[2];
        const auto token_pos=dims.d[1];
        const auto last_abs_token_offest=(token_pos-1)*v_s;
        ctx.next_token=ctx.sample->SampleTemperature(
            logits_address,
            last_abs_token_offest,
            v_s
        );
        ctx.abs_token_pos+=dims.d[1];
        // std::cout<<"ctx.next_token="<<ctx.next_token<<std::endl;
        // std::cout<<"ctx.abs_token_pos="<<ctx.abs_token_pos<<std::endl;
    }
    return code::operation::SUCCESS;
}
