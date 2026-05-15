//
// Created by HP on 2026/5/9.
//
#include "gdlz/ort/llm/onnx_rt_llm_framework.h"

#include <iostream>
#include <nlohmann/detail/exceptions.hpp>


auto getCtxInfo(const gdlz::ort::llm::data::OnnxRtLLmCtx& ctx) {

      std::cout<<"ctx.token_pos = "<<ctx.token_pos<<std::endl;
      std::cout<<"ctx.next_token_id = "<<ctx.next_token_id<<std::endl;
}
auto getInput(const std::vector<int64_t>& token_ids) {
    std::cout<<"print token_ids:"<<std::endl;
    for(auto& id:token_ids) {
        std::cout<<id<<std::endl;
    }
    std::cout<<"print tokens"<<std::endl;
}




int gdlz::ort::llm::OnnxRtLLmFramework::GetLayerNames(ort::data::OnnxRTEngine& engine, data::OnnxRtLLmCtx& ctx)
{

    return 0;
}

int gdlz::ort::llm::OnnxRtLLmFramework::CreateLLmEngine(
    data::OnnxRtLLmEngine& LlmEngine,
    ort::data::OnnxRTEngine& CoreEngine, const ort::data::OnnxRTEngineInfo& engine_info)
{

    LlmEngine.core_engine = std::make_shared<ort::data::OnnxRTEngine>(std::move(CoreEngine));
    LlmEngine.head_dim   = engine_info.head_dim;
    LlmEngine.layers     = engine_info.layers;
    LlmEngine.heads      = engine_info.heads;
    return 0;
}



int gdlz::ort::llm::OnnxRtLLmFramework::InitBatch(
    data::OnnxRtLLmEngine& engine,
    data::OnnxRtLLmCtx& ctx, data::OnnxRtLLmKv& kv, void* tokens,int len
) {

    try {

        ctx.next_token_id=0;
        if (kv.kv != nullptr) {
            kv.kv->clear();
        }
        auto* token_ids = static_cast<int64_t*>(tokens);
        if (ctx.memory_ptr == nullptr) {
            ctx.memory_ptr = std::make_unique<std::vector<int64_t>>();
        }
        ctx.memory_ptr->insert(ctx.memory_ptr->end(), token_ids, token_ids+len);
        std::cout<<"ctx.memory_ptr->size() = "<<ctx.memory_ptr->size()<<std::endl;
        std::vector<int64_t> input_ids =*ctx.memory_ptr;
        std::vector<int64_t> attention_mask(input_ids.size(), 1);
        std::vector<int64_t> positions     (input_ids.size());
        if (ctx.token_pos == 0){
             for (int64_t i = 0; i < input_ids.size(); ++i) positions[i] = i; //绝对位置
        }else{
             for (int64_t i = 0; i < input_ids.size(); ++i) positions[i] = ctx.token_pos + i; //相对位置
        }

        ctx.token_pos += static_cast<int64_t>(input_ids.size());

        std::vector<const char*> input_names;
        std::vector<const char*> output_names;

        for(auto& name: *(engine.core_engine->input_name_ptr ))  input_names.push_back (name.c_str());
        for(auto& name: *(engine.core_engine->output_name_ptr))  output_names.push_back(name.c_str());

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

        std::vector<int64_t> past_shape= {ctx.batch, engine.heads, 0, engine.head_dim};


        std::vector<Ort::Value> inputs;
        inputs.reserve(3+ engine.heads * engine.layers );
        inputs.push_back(std::move(input_ids_tensor));
        inputs.push_back(std::move(attn_mask_tensor));
        inputs.push_back(std::move(pos_ids_tensor));

        std::vector<Ort::Float16_t> empty_data(0);
        for (int i = 0; i != engine.layers; ++i) {

            auto key_tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
                           mem_info, empty_data.data(), empty_data.size(),
                           past_shape.data(), past_shape.size());

            auto value_tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
                    mem_info, empty_data.data(), empty_data.size(),
                    past_shape.data(), past_shape.size());

            inputs.push_back(std::move(key_tensor));
            inputs.push_back(std::move(value_tensor));
        }



        auto outputs =engine.core_engine->session->Run(
                  Ort::RunOptions{nullptr},
                  input_names.data(), inputs.data(), inputs.size(),
                   output_names.data(), output_names.size()
              );
        if (outputs.empty()) {
            return -1;
        }

        const auto& logits = outputs[0];
        const auto shape = logits.GetTensorTypeAndShapeInfo().GetShape();
        auto* logits_data = logits.GetTensorData<float>();
        constexpr int64_t batch_idx = 0;
        const int64_t last_pos = shape[1] - 1;
        const int64_t vocab_size = shape[2];

        const float* last_token_logits = logits_data + (batch_idx * shape[1] + last_pos) * vocab_size;
        int64_t next_token_id = std::distance(
            last_token_logits,
            std::max_element(last_token_logits, last_token_logits + vocab_size)
        );

        ctx.next_token_id = next_token_id;
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

    }catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

}

int gdlz::ort::llm::OnnxRtLLmFramework::GenerateToken(data::OnnxRtLLmEngine& engine, data::OnnxRtLLmCtx& ctx,
    data::OnnxRtLLmKv& kv)
{
    try
    {
        if (ctx.token_pos == 0 || kv.kv==nullptr){
            return -1;
        }
        std::vector<int64_t> input_ids={ctx.next_token_id};
        std::vector<int64_t> attention_mask={1};
        std::vector<int64_t> positions = {ctx.token_pos};
        ctx.token_pos++;
        std::vector<const char*> input_names;
        std::vector<const char*> output_names;

        for(auto& name: *(engine.core_engine->input_name_ptr ))  input_names.push_back (name.c_str());
        for(auto& name: *(engine.core_engine->output_name_ptr))  output_names.push_back(name.c_str());

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

        std::vector<int64_t> past_shape= {ctx.batch, engine.heads, 0, engine.head_dim};


        std::vector<Ort::Value> inputs;
        inputs.reserve(3+ engine.heads * engine.layers );
        inputs.push_back(std::move(input_ids_tensor));
        inputs.push_back(std::move(attn_mask_tensor));
        inputs.push_back(std::move(pos_ids_tensor));
        for (auto& cac: *(kv.kv)) {
            inputs.push_back(std::move(cac));
        }
        auto outputs =engine.core_engine->session->Run(
                 Ort::RunOptions{nullptr},
                 input_names.data(), inputs.data(), inputs.size(),
                  output_names.data(), output_names.size()
             );
        if (outputs.empty()) {
            return -1;
        }
        const auto& logits = outputs[0];
        const auto shape = logits.GetTensorTypeAndShapeInfo().GetShape();
        auto* logits_data = logits.GetTensorData<float>();
        constexpr int64_t batch_idx = 0;
        const int64_t last_pos = shape[1] - 1;
        const int64_t vocab_size = shape[2];

        const float* last_token_logits = logits_data + (batch_idx * shape[1] + last_pos) * vocab_size;
        int64_t next_token_id = std::distance(
            last_token_logits,
            std::max_element(last_token_logits, last_token_logits + vocab_size)
        );
        ctx.next_token_id = next_token_id;
        if (ctx.memory_ptr != nullptr) {
            ctx.memory_ptr->push_back(next_token_id);
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
