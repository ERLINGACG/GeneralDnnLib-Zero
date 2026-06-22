//
// Created by HP on 2026/5/9.
//
#include "gdlz/ort/llm/onnx_rt_llm_framework.h"

#include <cstring>
#include <iostream>
#include <dylog/logger.h>
#include <nlohmann/detail/exceptions.hpp>

#include "gdlz/ort/core/onnx_rt_framework.h"
#include "gdlz/ort/llm/onnx_rt_llm_director.h"




gdlz::ort::llm::OnnxRtLLmFramework::OnnxRtLLmFramework(const char* config_path)
{

    auto  json_config=nlohmann::json();
    OnnxRtLLmDirector::Handle(engine_info, config_path,json_config);
    OnnxRtLLmDirector::CreateEngine(engine,engine_info,json_config);
}


void gdlz::ort::llm::OnnxRtLLmFramework::GetLayerInfo()
{
    core::OnnxRtFramework::GetEngineInfo(engine.core_engine);
}

int gdlz::ort::llm::OnnxRtLLmFramework::InitSampler(sampler::OnnxRtLLmParam& param, data::OnnxRtLLmCtx& ctx)
{
    ctx.sampler=std::make_unique<sampler::OnnxRtLLmSampler>();
    ctx.sampler->SetParam(param);
    return 0;
}

int gdlz::ort::llm::OnnxRtLLmFramework::PrefillFor1DRoPE(data::OnnxRtLLmCtx& ctx, data::OnnxRtLLmKv& kv, void* tokens,
    int64_t len) const {

    return this->InitBatchForTokenIds(ctx,kv,tokens,len);
}

int gdlz::ort::llm::OnnxRtLLmFramework::DecodeFor1DRoPE(data::OnnxRtLLmCtx& ctx, data::OnnxRtLLmKv& kv) const {

    return this->GenerateToken(ctx,kv);
}


int gdlz::ort::llm::OnnxRtLLmFramework::InitBatchForTokenIds(
    data::OnnxRtLLmCtx& ctx,
    data::OnnxRtLLmKv& kv,
    void* tokens,int64_t len
) const {

    try
    {
        ctx.next_token_id=0;
        if (kv.kv==nullptr) {
            kv.kv = std::make_unique<std::vector<Ort::Value>>();
        }

        if (kv.KvBuffer==nullptr){
            kv.KvBuffer=std::make_unique<std::vector<std::vector<uint8_t>>>();
        }


        if (kv.prefixKvBuffer == nullptr) {
            kv.prefixKvBuffer=std::make_unique<std::vector<std::vector<uint8_t>>>();
        }

        if (kv.prefixKv==nullptr){
            kv.prefixKv=std::make_unique<std::vector<Ort::Value>>();
        }

        auto* token_ids = static_cast<int64_t*>(tokens);
        if (ctx.memory_ptr == nullptr) {
            ctx.memory_ptr = std::make_unique<std::vector<int64_t>>();
        }

        auto clear=[&](){
            kv.prefixKv->clear();
            kv.prefixKvBuffer->clear();

        };

        auto copy_decode_kv=[&](){
             for (auto& item:*kv.kv){

                 auto info=item.GetTensorTypeAndShapeInfo();
                 auto shape=info.GetShape();
                 auto type=info.GetElementType();
                 int64_t size=1;
                 for (auto dim: shape){size *= dim;}
                 if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT){

                     auto* data=item.GetTensorData<float>();
                     auto* begin = reinterpret_cast<const uint8_t*>(data);
                     auto* end = begin + sizeof(float) * size;
                     kv.prefixKvBuffer->emplace_back(begin,end);
                     auto& back_data=kv.prefixKvBuffer->back();
                     auto tensor=Ort::Value::CreateTensor<float>(
                                 mem_info, reinterpret_cast<float*>(back_data.data()), size,
                                 shape.data(), shape.size()
                     );
                     kv.prefixKv->push_back(std::move(tensor));

                 }
             }
             kv.kv->clear();
             kv.KvBuffer->clear();
        };

        clear();
        copy_decode_kv();

        std::cout<<"history:"<<ctx.memory_ptr->size()<<std::endl;
        std::vector<int64_t> input_ids={token_ids, token_ids+len};
        int index=0;
        for (auto item: *ctx.memory_ptr){
            if (index >= input_ids.size()){
                break;
            }
            if (item == input_ids[index]){
                 ++index;
            }else {
                break;
            }
        }
        std::cout<<"index:"<<index<<std::endl;
        // if (index > 0){
        //     input_ids.erase(input_ids.begin(), input_ids.begin()+index);
        // }
        ctx.memory_ptr->insert(ctx.memory_ptr->end(), token_ids, token_ids+len);
        std::vector<int64_t> positions     (input_ids.size());

        if (ctx.token_pos == 0){
            for (int64_t i = 0; i < input_ids.size(); ++i) positions[i] = i; //绝对位置
        }else{
            for (int64_t i = 0; i < input_ids.size(); ++i) positions[i] = ctx.token_pos + i; //相对位置
        }
        // std::cout<<"token_pos:"<<ctx.token_pos<<std::endl;

        std::cout<<"positions:[";
        for (auto item: positions){
            std::cout<<item<<",";
        }
        std::cout<<"]"<<std::endl;
        std::vector<int64_t> attention_mask(input_ids.size()+ctx.memory_ptr->size(), 1);
        std::cout<<"attention_mask:[";
        for (auto item: attention_mask){
            std::cout<<item<<",";
        }
        std::cout<<"]"<<std::endl;
        ctx.token_pos += static_cast<int64_t>(input_ids.size());


        std::vector<const char*> input_names;
        std::vector<const char*> output_names;

        for(auto& name: *(engine.core_engine.input_name_ptr ))  input_names.push_back (name.c_str());
        for(auto& name: *(engine.core_engine.output_name_ptr))  output_names.push_back(name.c_str());


        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector input_shape = {ctx.batch, static_cast<int64_t>(input_ids.size())};

        auto input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
                mem_info, input_ids.data(), input_ids.size(),
                input_shape.data(), input_shape.size());

        auto attn_mask_tensor = Ort::Value::CreateTensor<int64_t>(
            mem_info, attention_mask.data(), attention_mask.size(),
            input_shape.data(), input_shape.size());

        auto pos_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            mem_info, positions.data(), positions.size(),
            input_shape.data(), input_shape.size());


        std::vector<Ort::Value> inputs;
        inputs.reserve(engine.layers);
        for (auto& item: engine.input_shapes){
            if (item.type=="input"){;
                inputs.insert(inputs.begin()+item.layer_pos, std::move(input_ids_tensor));
            }
            if (item.type=="mask") {
                inputs.insert(inputs.begin()+item.layer_pos, std::move(attn_mask_tensor));
            }
            if (item.type=="pos") {
                inputs.insert(inputs.begin()+item.layer_pos, std::move(pos_ids_tensor));
            }

            if (item.type=="kv_cache" || item.type=="start_kv_cache") {
                if (item.data_type=="float16") {
                        for (int i=0; i<item.num; i++) {
                            auto empty_data = std::vector<Ort::Float16_t>();
                            auto tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
                                mem_info, empty_data.data(), empty_data.size(),
                                item.shape.data(), item.shape.size()
                            );
                            inputs.insert(inputs.begin()+item.layer_pos+i, std::move(tensor));
                        }
                }
                if (item.data_type=="float32") {
                    for (int i=0; i<item.num; i++) {
                        auto empty_data = std::vector<float>();
                        auto tensor = Ort::Value::CreateTensor<float>(
                                mem_info, empty_data.data(), empty_data.size(),
                                item.shape.data(), item.shape.size()
                        );
                        inputs.insert(inputs.begin()+item.layer_pos+i, std::move(tensor));
                    }
                }
            }
        }

        auto outputs =engine.core_engine.session->Run(
                  Ort::RunOptions{nullptr},
                  input_names.data(), inputs.data(), inputs.size(),
                   output_names.data(), output_names.size()
         );

        if (outputs.empty()) {
            return -1;
        }

        ctx.next_token_id = ctx.sampler->SampleToken(outputs);

        ctx.memory_ptr->push_back(ctx.next_token_id);
        std::vector<Ort::Value> next_past_kv;
        next_past_kv.reserve(outputs.size() - 1);

        for (size_t i = 1; i < outputs.size(); ++i) {
            next_past_kv.push_back(std::move(outputs[i]));
        }

        auto save_kv_test=[&]() {

            for (auto& item: next_past_kv){
                auto tensor_info = item.GetTensorTypeAndShapeInfo();
                auto shape = tensor_info.GetShape();
                auto data = item.GetTensorData<float>();
                int64_t size = 1;
                for (const auto dim : shape) {
                    size *= dim;
                }
                const auto* begin = reinterpret_cast<const uint8_t*>(data);
                const auto* end = begin + size * sizeof(float);
                kv.prefixKvBuffer->emplace_back(begin, end);
                auto& buffer = kv.prefixKvBuffer->back();

                auto tensor = Ort::Value::CreateTensor<float>(
                        mem_info, reinterpret_cast<float*>(buffer.data()) , size,
                        shape.data(), shape.size());
                kv.prefixKv->push_back(std::move(tensor));
            }
            // if (kv.prefixKv->size()> 56*3 && kv.prefixKvBuffer->size()> 56*3){
            //     kv.prefixKv->erase(kv.prefixKv->begin(), kv.prefixKv->begin()+56);
            //     kv.prefixKvBuffer->erase(kv.prefixKvBuffer->begin(), kv.prefixKvBuffer->begin()+56);
            // }
        };
        auto expand_kv_test=[&](){

            if (kv.prefixKv->size() % 56 == 0 && kv.prefixKv->size() != 56){
                next_past_kv.clear();
                for (int i=0; i<56; i++) {

                    if (i+56>=kv.prefixKv->size())  return;
                    auto kv_1_shape=kv.prefixKv->at(i).GetTensorTypeAndShapeInfo().GetShape();
                    auto kv_1_data=kv.prefixKv->at(i).GetTensorData<float>();
                 //   kv_1_shape[2]=index;


                    auto kv_2_shape=kv.prefixKv->at(i+56).GetTensorTypeAndShapeInfo().GetShape();
                    auto kv_2_data=kv.prefixKv->at(i+56).GetTensorData<float>();

                    int64_t size_1 = 1;
                    std::cout<<"kv_1_shape：[ ";
                    for (auto dim : kv_1_shape) {size_1*=dim;std::cout<<" "<<dim;}
                    std::cout<<" ]";

                    int64_t size_2=1;
                    std::cout<<" kv_2_shape?[ ";
                    for (auto dim : kv_2_shape) {size_2*=dim;std::cout<<" "<<dim;}
                    std::cout<<" ]"<<std::endl;

                    constexpr int64_t batch = 1;
                    constexpr int64_t heads = 4;
                    constexpr int64_t head_dim = 128;
                    const int64_t seq_1 = kv_1_shape[2];
                    const int64_t seq_2 = kv_2_shape[2];
                    std::vector<int64_t> shape={batch, heads, seq_1 + seq_2, head_dim};

                    const size_t old_head_elems = static_cast<size_t>(seq_1 * head_dim);
                    const size_t new_head_elems = static_cast<size_t>(seq_2 * head_dim);
                    const size_t merged_head_elems = static_cast<size_t>((seq_1 + seq_2) * head_dim);

                    std::vector<uint8_t> merged(
                        static_cast<size_t>(batch * heads) * merged_head_elems * sizeof(float)
                    );

                    auto* merged_data = reinterpret_cast<float*>(merged.data());
                    for (int64_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
                        for (int64_t head_idx = 0; head_idx < heads; ++head_idx) {
                            const size_t old_offset =
                                static_cast<size_t>(batch_idx * heads + head_idx) * old_head_elems;
                            const size_t new_offset =
                                static_cast<size_t>(batch_idx * heads + head_idx) * new_head_elems;
                            const size_t merged_offset =
                                static_cast<size_t>(batch_idx * heads + head_idx) * merged_head_elems;

                            std::memcpy(
                                merged_data + merged_offset,
                                kv_1_data + old_offset,
                                old_head_elems * sizeof(float)
                            );
                            std::memcpy(
                                merged_data + merged_offset + old_head_elems,
                                kv_2_data + new_offset,
                                new_head_elems * sizeof(float)
                            );
                        }
                    }
                    kv.KvBuffer->emplace_back(std::move(merged));
                    auto& data=kv.KvBuffer->back();
                    size_t float_count = data.size() / sizeof(float);
                    int64_t size = 1;
                    for (auto dim: shape){ size *= dim; }
                    std::cout<<"size:"<<size<<" float_count:"<<float_count<<std::endl;
                    auto tensor = Ort::Value::CreateTensor<float>(
                        mem_info, reinterpret_cast<float*>(data.data()) ,  float_count,
                        shape.data(), shape.size()
                    );

                    next_past_kv.push_back(std::move(tensor));

                }
            }


        };

        auto expand_kv=[&](){
            if (!kv.prefixKv->empty() && kv.prefixKv->size()==next_past_kv.size()){
                int _index=0;
                for (auto& item: *(kv.prefixKv)){

                    auto item_info             =item.GetTensorTypeAndShapeInfo();
                    auto item_shape=item_info.GetShape();
                    auto item_type=item_info.GetElementType();

                    auto next_kv_info=next_past_kv.at(_index).GetTensorTypeAndShapeInfo();
                    auto next_kv_shape=next_kv_info.GetShape();

                    auto batch    =item_shape[0];
                    auto heads    =item_shape[1];
                    auto head_dim =item_shape[3];

                    auto seq_1=item_shape[2];
                    auto seq_2=next_kv_shape[2];
                    std::vector<int64_t> shape={batch, heads, seq_1 + seq_2, head_dim};
                    int64_t item_size=1;
                    int64_t next_kv_size=1;
                    const auto old_head_elems = static_cast<size_t>(seq_1 * head_dim);
                    const auto new_head_elems = static_cast<size_t>(seq_2 * head_dim);
                    const auto merged_head_elems = static_cast<size_t>((seq_1 + seq_2) * head_dim);

                    for (auto dim : item_shape)   {item_size*=dim;}
                    for (auto dim : next_kv_shape){next_kv_size*=dim;}

                    if (item_type==ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                        auto item_data   =item.GetTensorData<float>();
                        auto next_kv_data=next_past_kv.at(_index).GetTensorData<float>();

                        std::vector<uint8_t> merged(static_cast<size_t>(batch * heads) * merged_head_elems * sizeof(float));
                        auto* merged_data = reinterpret_cast<float*>(merged.data());
                        for (int64_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
                            for (int64_t head_idx = 0; head_idx < heads; ++head_idx) {
                                const size_t old_offset =
                                    static_cast<size_t>(batch_idx * heads + head_idx) * old_head_elems;
                                const size_t new_offset =
                                    static_cast<size_t>(batch_idx * heads + head_idx) * new_head_elems;
                                const size_t merged_offset =
                                    static_cast<size_t>(batch_idx * heads + head_idx) * merged_head_elems;

                                std::memcpy(
                                    merged_data + merged_offset,
                                    item_data + old_offset,
                                    old_head_elems * sizeof(float)
                                );
                                std::memcpy(
                                    merged_data + merged_offset + old_head_elems,
                                    next_kv_data + new_offset,
                                    new_head_elems * sizeof(float)
                                );
                            }
                        }
                        kv.KvBuffer->emplace_back(std::move(merged));
                        auto& data=kv.KvBuffer->back();
                        size_t float_count = data.size() / sizeof(float);
                        int64_t size = 1;
                        for (auto dim: shape){ size *= dim; }

                        auto tensor = Ort::Value::CreateTensor<float>(
                            mem_info, reinterpret_cast<float*>(data.data()) ,  float_count,
                            shape.data(), shape.size()
                        );
                        kv.kv->push_back(std::move(tensor));
                    };

                    _index++;

                }
                next_past_kv.clear();
            }else{
                *(kv.kv) = std::move(next_past_kv);
            }
        };

        // save_kv();
        expand_kv();
        std::cout<<"prefix size:"<<kv.prefixKv->size()<<std::endl;
        std::cout<<"nex_past_kv:"<<next_past_kv.size()<<std::endl;


        return 0;

    }catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

}

int gdlz::ort::llm::OnnxRtLLmFramework::GenerateToken(
    data::OnnxRtLLmCtx& ctx,
    data::OnnxRtLLmKv& kv
) const {
    try {
        if (ctx.token_pos == 0 || kv.kv==nullptr){
            return -1;
        }
        std::vector<int64_t> input_ids={ctx.next_token_id};
        std::vector<int64_t> attention_mask={1};
        std::vector<int64_t> positions = {ctx.token_pos};
        ctx.token_pos++;
        std::vector<const char*> input_names;
        std::vector<const char*> output_names;

        for(auto& name: *(engine.core_engine.input_name_ptr ))  input_names.push_back (name.c_str());
        for(auto& name: *(engine.core_engine.output_name_ptr))  output_names.push_back(name.c_str());

        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector input_shape = {ctx.batch, static_cast<int64_t>(input_ids.size())};

        auto input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
              mem_info, input_ids.data(), input_ids.size(),
              input_shape.data(), input_shape.size());

        auto attn_mask_tensor = Ort::Value::CreateTensor<int64_t>(
            mem_info, attention_mask.data(), attention_mask.size(),
            input_shape.data(), input_shape.size());

        auto pos_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            mem_info, positions.data(), positions.size(),
            input_shape.data(), input_shape.size());


        std::vector<Ort::Value> inputs;
        inputs.reserve(engine.layers);
        int kv_start_pos=0;
        for (auto& item: engine.input_shapes){
            if (item.type=="input") {
                inputs.insert(inputs.begin()+item.layer_pos, std::move(input_ids_tensor));
            }
            if (item.type=="mask") {
                inputs.insert(inputs.begin()+item.layer_pos, std::move(attn_mask_tensor));
            }
            if (item.type=="pos") {
                inputs.insert(inputs.begin()+item.layer_pos, std::move(pos_ids_tensor));
            }
            if (item.type=="start_kv_cache"){
                kv_start_pos=item.layer_pos;
            }
        }
        for (auto& cac: *(kv.kv)) {
            inputs.insert(inputs.begin()+kv_start_pos, std::move(cac));
            kv_start_pos++;
        }

        auto outputs =engine.core_engine.session->Run(
                 Ort::RunOptions{nullptr},
                 input_names.data(), inputs.data(), inputs.size(),
                  output_names.data(), output_names.size()
             );
        if (outputs.empty()) {
            return -1;
        }


        ctx.next_token_id= ctx.sampler->SampleToken(outputs);
        if (ctx.memory_ptr != nullptr ) {
            ctx.memory_ptr->push_back(ctx.next_token_id);
        }
        if (kv.kv==nullptr) {
            kv.kv = std::make_unique<std::vector<Ort::Value>>();
        }
        std::vector<Ort::Value> next_past_kv;
        next_past_kv.reserve(outputs.size() - 1);
        for (size_t i = 1; i < outputs.size(); ++i) {
            next_past_kv.push_back(std::move(outputs[i]));
        }
        *(kv.kv) = std::move(next_past_kv);

        return 0;
    }catch (std::exception& e){
        std::cerr << e.what() << std::endl;
        return -1;
    }
}
