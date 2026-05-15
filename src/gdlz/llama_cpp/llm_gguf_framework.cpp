//
// Created by HP on 2026/1/27.
//
#include "gdlz//llama/framework/llm_gguf_framework.h"

#include <iostream>
#include <unordered_set>
#include <vector>
#include <windows.h>

#undef ERROR
#include <dylog/logger.h>
#pragma pop_macro("ERROR")

#include <opencv2/core.hpp>
using gdlz::llm::gguf::LLm_GGuf_Framework;

LLm_GGuf_Framework& LLm_GGuf_Framework::ResourceDirector(
    resources::llm::gguf::LLm_GGuf_ResourceDirector& director
    )
{
    director.SetConfPath(this->conf_path).Hand(this->resource);
    return *this;
}

void LLm_GGuf_Framework::releases()
{
    resource.release();
}

// void LLm_GGuf_Framework::ExampleSend() const
// {
//     std::string send_str="你好,请介绍你自己";
//     std::string prompt =
//             "<|im_start|>user\n"          // 用户角色开始
//             + send_str +"\n"             // 提问（换行分隔，去掉多余空格）
//             "<|im_end|>\n"                // 用户角色结束
//             "<|im_start|>assistant\n";    // 助手角色开始（关键：结尾不加<|im_end|>，留给模型生成）
//     auto *tokens = static_cast<llama_token*>(malloc(sizeof(llama_token) * 10000));
//     int len = llama_tokenize(
//                        resource.vocab, prompt.c_str(),
//                        static_cast<int32_t>(prompt.size()),
//                        tokens, 10000,
//                        false, true
//                        );
//     if (len < 0){
//         return;
//     }
//     llama_batch batch = llama_batch_get_one(tokens, len);
//     llama_token next_token = LLAMA_TOKEN_NULL;
//     llama_token eos = llama_vocab_eos(resource.vocab);
//
//     while (next_token != eos){
//             if(llama_decode(resource.context, batch)){
//                 std::cerr << "Error: decode error" << std::endl;
//                 return;
//             }
//             next_token = llama_sampler_sample(resource.sampler, resource.context, -1);
//             char de_prompt[100] = {0};
//             if(llama_detokenize(resource.vocab, &next_token, 1, de_prompt, sizeof(de_prompt) / sizeof(de_prompt[0]), false, false) < 0){
//                 std::cerr << "Error: detokenize error" << std::endl;
//                 return ;
//             }
//
//             std::cout << "deprompt:" << de_prompt << std::endl;
//             batch.token[0] = next_token;
//             batch.n_tokens = 1;
//
//     }
//
//     llama_batch_free(batch);
//
// }

// int32_t LLm_GGuf_Framework::SendExample(
//     const char* prompts,
//     batch::LLM_GGUF_Batch& batch, data::LLM_GGUF_Stream& stream
//     ) const
// {
//     // std::string send_str="你好,请介绍你自己";
//     std::string prompt =
//             "<|im_start|>user\n"          // 用户角色开始
//             + std::string(prompts) +"\n"             // 提问（换行分隔，去掉多余空格）
//             "<|im_end|>\n"                // 用户角色结束
//             "<|im_start|>assistant\n";    // 助手角色开始（关键：结尾不加<|im_end|>，留给模型生成）
//
//     auto *tokens = static_cast<llama_token*>(malloc(sizeof(llama_token) * 10000));
//     int len = llama_tokenize(
//                      resource.vocab, prompt.c_str(),
//                      static_cast<int32_t>(std::strlen(prompt.c_str())),
//                      tokens, 10000,
//                      false, true
//                      );
//     batch.batch = std::make_unique<llama_batch>(llama_batch_get_one(tokens, len));
//     batch.eos    = llama_vocab_eos(resource.vocab);
//
//
//     if (batch.next_token != batch.eos && !llama_decode(resource.context, *batch.batch)) {
//         batch.next_token = llama_sampler_sample(resource.sampler, resource.context, -1);
//         std::cout << "next_token:" << batch.next_token << std::endl;
//         auto res= llama_detokenize(
//             resource.vocab, &batch.next_token, 1, stream.stream,
//             static_cast<int>(std::size(stream.stream)),
//             false,
//             false);
//         std::cout << "de_prompt:" << stream.stream <<" len:"<<  std::strlen(stream.stream) << std::endl;
//         stream.str_len = static_cast<int>(std::strlen(stream.stream));
//         //     ) < 0;
//         return res;
//     }
//     return 0;
// }

// void LLm_GGuf_Framework::SetSamplerRT(
//     const char* gbnf_str,
//     const bool use_grammar,
//     const int top_k,
//     const float top_p,
//     const float temp
// ) const {
//     try{
//         if (resource.sampler != nullptr) {
//             llama_sampler_free(resource.sampler);
//         }
//         resource.sampler=llama_sampler_chain_init(resource.sampler_chain_params);
//         if (use_grammar) {
//             llama_sampler* grammar_sampler = llama_sampler_init_grammar(
//             resource.vocab,          // 模型词汇表
//             gbnf_str,// GBNF规则字符串
//              "root"          // 根规则名
//              );
//             if (grammar_sampler != nullptr){
//                 llama_sampler_chain_add(resource.sampler, grammar_sampler);
//             }
//         }
//         llama_sampler_chain_add(resource.sampler, llama_sampler_init_temp(temp));
//         llama_sampler_chain_add(resource.sampler, llama_sampler_init_top_k(top_k));
//         llama_sampler_chain_add(resource.sampler, llama_sampler_init_top_p(top_p, 1));
//         llama_sampler_chain_add(resource.sampler, llama_sampler_init_dist(time(nullptr)));
//         llama_sampler_chain_add(resource.sampler, llama_sampler_init_penalties(300,1.2,0.8,1.5));
//
//     }catch (std::exception& e) {
//         std::cerr << e.what() << std::endl;
//     }
// }
//
// void LLm_GGuf_Framework::InitBatch(const char* prompt, batch::LLM_GGUF_Batch& batch) const
// {
//     std::string prompts =std::string(prompt);
//     auto *tokens = static_cast<llama_token*>(malloc(sizeof(llama_token) * 10000));
//     int len = llama_tokenize(
//                      resource.vocab, prompts.c_str(),
//                      static_cast<int32_t>(std::strlen(prompts.c_str())),
//                      tokens, 10000,
//                      false, true
//                      );
//     batch.batch = std::make_unique<llama_batch>(llama_batch_get_one(tokens, len));
//     batch.eos    = llama_vocab_eos(resource.vocab);
//
// }
//
// void LLm_GGuf_Framework::Reasoning(batch::LLM_GGUF_Batch& batch, data::LLM_GGUF_Stream& stream) const
// {
//     if (batch.next_token != batch.eos && !llama_decode(resource.context, *batch.batch)) {
//
//         batch.next_token = llama_sampler_sample(resource.sampler, resource.context, -1);
//         llama_detokenize(
//             resource.vocab, &batch.next_token, 1, stream.stream,
//             static_cast<int>(std::size(stream.stream)),
//             false,
//             false);
//         stream.str_len = static_cast<int>(std::strlen(stream.stream));
//         batch.batch->token[0] = batch.next_token;
//         batch.batch->n_tokens = 1;
//         //     ) < 0;
//     }
// }

void LLm_GGuf_Framework::SetSamplerASync(
    const batch::LLM_GGUF_Context_RTParam& param,
    batch::LLM_GGUF_Context& context
) const
{
       auto debug=[&]()
       {
           std::cout << "cpp_get_top_k=" << param.top_k << std::endl;
           std::cout << "cpp_get_top_p=" << param.top_p << std::endl;
           std::cout << "cpp_get_temp=" << param.temp << std::endl;
           std::cout << "cpp_get_n_batch=" << param.n_batch << std::endl;
           std::cout << "cpp_get_n_ctx=" << param.n_ctx << std::endl;
           std::cout << "cpp_get_use_gbnf=" << param.use_gbnf << std::endl;
           std::cout << "cpp_get_use_embeddings=" << param.use_embeddings << std::endl;
           std::cout << "cpp_get_penalty_last_n=" << param.penalty_last_n << std::endl;
           std::cout << "cpp_get_penalty_repeat=" << param.penalty_repeat << std::endl;
           std::cout << "cpp_get_penalty_freq=" << param.penalty_freq << std::endl;

       };

        // debug();
       if (context.context==nullptr){
           llama_context_params context_params=llama_context_default_params();
            context_params.n_ctx  =param.n_ctx;
            context_params.n_batch=param.n_batch;
            context_params.embeddings=param.use_embeddings;
            if (context_params.embeddings)
            {
                std::cout << "embeddings=" << context_params.embeddings << std::endl;
            }
            context_params.pooling_type = LLAMA_POOLING_TYPE_MEAN;
            context.context=llama_init_from_model(resource.model,context_params);
       }

       if (context.sampler!=nullptr){
           llama_sampler_free(context.sampler);
       }

       context.sampler=llama_sampler_chain_init(llama_sampler_chain_default_params());
       if (param.use_gbnf && context.sampler!=nullptr){
           llama_sampler* grammar_sampler = llama_sampler_init_grammar(
            resource.vocab,          // 模型词汇表
            param.gbnf_str,         // GBNF规则字符串
             "root"                 // 根规则名
             );
            if (grammar_sampler != nullptr){
                llama_sampler_chain_add(context.sampler, grammar_sampler);
            }
       }

       if (context.sampler!=nullptr){
           llama_sampler_chain_add(context.sampler, llama_sampler_init_temp(param.temp));
           llama_sampler_chain_add(context.sampler, llama_sampler_init_top_k(param.top_k));
           llama_sampler_chain_add(context.sampler, llama_sampler_init_top_p(param.top_p, 1));
           llama_sampler_chain_add(context.sampler, llama_sampler_init_dist(time(nullptr)));
           llama_sampler_chain_add(context.sampler,
               llama_sampler_init_penalties(
                   param.penalty_last_n,
                   param.penalty_repeat,
                   param.penalty_freq,
                   param.penalty_present
               )
           );
       }

}

int  LLm_GGuf_Framework::InitBatchASync(
    const char* prompt,
    batch::LLM_GGUF_Batch& batch,
    batch::LLM_GGUF_Context& context
) const
{
    std::string prompts =std::string(prompt);
    auto *tokens = static_cast<llama_token*>(malloc(sizeof(llama_token) * prompts.size() * 3 ));
    int len = llama_tokenize(
                     resource.vocab, prompts.c_str(),
                     static_cast<int32_t>(prompts.size()),
                     tokens, static_cast<int32_t>(prompts.size() * 3),
                     false, true
    );
    batch.batch = std::make_unique<llama_batch>(llama_batch_get_one(tokens, len));
    batch.eos    = llama_vocab_eos(resource.vocab);
    return len;

}

void LLm_GGuf_Framework::ReasoningASync(
    batch::LLM_GGUF_Batch& batch,
    batch::LLM_GGUF_Context& context,
    data::LLM_GGUF_Stream& stream
) const
{
    if (batch.next_token==batch.eos){
        return;
    }
    if (!llama_decode(context.context, *batch.batch)) {
        stream.stream[0]='\0';
        stream.str_len=0;
        batch.next_token = llama_sampler_sample(context.sampler, context.context, -1);

        llama_detokenize(
            resource.vocab, &batch.next_token, 1, stream.stream,
            static_cast<int32_t>(std::size(stream.stream)-1),
            false,
            false
        );
        stream.str_len = static_cast<int>(std::strlen(stream.stream));
        batch.batch->token[0] = batch.next_token;
        batch.batch->n_tokens = 1;
    }
}

void LLm_GGuf_Framework::InitEmbeddings(
    const char* prompt,
    batch::LLM_GGUF_Batch& batch,
    batch::LLM_GGUF_Context& context
) const
{
    std::string prompts =std::string(prompt);
    auto *tokens = static_cast<llama_token*>(malloc(sizeof(llama_token) * prompts.size() * 3 ));
    int len = llama_tokenize(
                     resource.vocab, prompts.c_str(),
                     static_cast<int32_t>(prompts.size()),
                     tokens, static_cast<int32_t>(prompts.size() * 3),
                     false, true
    );
    batch.batch = std::make_unique<llama_batch>(llama_batch_get_one(tokens, len));
    batch.eos    = llama_vocab_eos(resource.vocab);
    context.keep_recent_tokens(0,2048);

}

void LLm_GGuf_Framework::InitEmbeddings_0(
    const char* prompt,
    batch::LLM_GGUF_Batch& batch,
    batch::LLM_GGUF_Context& context
) const
{
    std::string prompts =std::string(prompt);
    auto *tokens = static_cast<llama_token*>(malloc(sizeof(llama_token) * prompts.size() * 3 ));
    int len = llama_tokenize(
                     resource.vocab, prompts.c_str(),
                     static_cast<int32_t>(prompts.size()),
                     tokens, static_cast<int32_t>(prompts.size() * 3),
                     false, true
    );
    batch.batch = std::make_unique<llama_batch>(llama_batch_get_one(tokens, len));
    batch.eos    = llama_vocab_eos(resource.vocab);
    // 执行解码
    if (llama_decode(context.context, *batch.batch) != 0) {
        free(tokens);
        throw std::runtime_error("llama_decode failed");
    }
    // 获取嵌入向量（取决于池化类型）
    const int n_embd_out = llama_model_n_embd(resource.model);
    const enum llama_pooling_type pooling_type = llama_pooling_type(context.context);

    if (pooling_type==LLAMA_POOLING_TYPE_MEAN)
    {
        const float * embd = nullptr;
        embd = llama_get_embeddings_seq(context.context, 0);   // 池化后的序列嵌入
        std::vector<float> vec(embd, embd + n_embd_out);
        cv::Mat emb_mat(1, n_embd_out, CV_32F, vec.data());
        cv::normalize(emb_mat, emb_mat, 1.0, 0.0, cv::NORM_L2);
        // std::cout << "Normalized embedding (first 10 of " << n_embd_out << "): ";
        // for (int i = 0; i < 10 && i < n_embd_out; ++i) {
        //     std::cout << vec[i] << " " <<std::endl;
        // }
        for (int i = 0; i < n_embd_out; i++)
        {
            std::cout << vec[i] << std::endl;
        }


        // for (int j = 0; j < n_embd_out; j++)
        // {
        //     std::cout << embd[j] << " ";
        // }
        // for (int i = 0; i < batch.batch->n_tokens; i++)
        // {
        //
        //     // int n_embd = llama_model_n_embd(resource.model); // 获取维度
        //
        //     if (embd != nullptr){
        //         for (int j = 0; j < n_embd_out; j++) {
        //                 std::cout << embd[j] << " "; // 打印每个浮点数
        //         }
        //         std::cout << embd << std::endl;
        //     }else{
        //         std::cout <<"Failed to get embeddings for token "<< i << std::endl;
        //     }
        // }
    }
    free(tokens);
    // std::cout << n_embd << std::endl;
    // const float* embeddings = nullptr;
    // embeddings = llama_get_embeddings_seq(resource.context, 0);  // 序列 ID 为 0
    // if (!embeddings) {
    //     std::cout << "Failed to get embeddings" << std::endl;
    //     return;
    // }

    //
    // for (const auto e : embedding){
    //      std::cout << e << std::endl;
    // }






}

void LLm_GGuf_Framework::GetEmbeddings(
    batch::LLM_GGUF_Batch& batch,
    batch::LLM_GGUF_Context& context,
    data::LLM_GGUF_Embedding& embedding
) const
{

    if (llama_decode(context.context, *batch.batch) != 0){
        throw std::runtime_error("llama_decode failed");
    }
    const int n_embd_out = llama_model_n_embd(resource.model);
    const enum llama_pooling_type pooling_type = llama_pooling_type(context.context);
    if (pooling_type==LLAMA_POOLING_TYPE_MEAN)
    {
        std::cout<<"pooling_type==LLAMA_POOLING_TYPE_MEAN"<<std::endl;
        float * embd = nullptr;
        embd = llama_get_embeddings_seq(context.context, 0);   // 池化后的序列嵌入
        std::vector vec(embd, embd + n_embd_out);

        dylog::DynamicLogger().setInvokeName("GetEmbeddings").info("n_embd_sizes={}", vec.size());
        for (int i = 0; i < 30; i++){
            dylog::DynamicLogger().info("embd[{}]={}", i, vec[i]);
        }
        embedding.embeddings = std::make_unique<float[]>(n_embd_out);
        std::memcpy(embedding.embeddings.get(), embd, n_embd_out * sizeof(float));
        embedding.embeddings_len = n_embd_out;
    }


}


