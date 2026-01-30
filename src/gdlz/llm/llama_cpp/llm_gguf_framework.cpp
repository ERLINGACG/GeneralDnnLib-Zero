//
// Created by HP on 2026/1/27.
//
#include "gdlz//llm/framework/llm_gguf_framework.h"

#include <iostream>
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

void LLm_GGuf_Framework::ExampleSend() const
{
    std::string send_str="你好,请介绍你自己";
    std::string prompt =
            "<|im_start|>user\n"          // 用户角色开始
            + send_str +"\n"             // 提问（换行分隔，去掉多余空格）
            "<|im_end|>\n"                // 用户角色结束
            "<|im_start|>assistant\n";    // 助手角色开始（关键：结尾不加<|im_end|>，留给模型生成）
    auto *tokens = static_cast<llama_token*>(malloc(sizeof(llama_token) * 10000));
    int len = llama_tokenize(
                       resource.vocab, prompt.c_str(),
                       static_cast<int32_t>(prompt.size()),
                       tokens, 10000,
                       false, true
                       );
    if (len < 0){
        return;
    }
    llama_batch batch = llama_batch_get_one(tokens, len);
    llama_token next_token = LLAMA_TOKEN_NULL;
    llama_token eos = llama_vocab_eos(resource.vocab);

    while (next_token != eos){
            if(llama_decode(resource.context, batch)){
                std::cerr << "Error: decode error" << std::endl;
                return;
            }
            next_token = llama_sampler_sample(resource.sampler, resource.context, -1);
            char de_prompt[100] = {0};
            if(llama_detokenize(resource.vocab, &next_token, 1, de_prompt, sizeof(de_prompt) / sizeof(de_prompt[0]), false, false) < 0){
                std::cerr << "Error: detokenize error" << std::endl;
                return ;
            }

            std::cout << "deprompt:" << de_prompt << std::endl;
            batch.token[0] = next_token;
            batch.n_tokens = 1;

    }

    llama_batch_free(batch);

}
