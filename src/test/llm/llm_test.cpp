//
// Created by HP on 2025/12/20.

#include <cstring>  // 补充memset头文件
#include "llama.h"  // 引入llama.cpp核心头文件
#include <iostream>
#include <vector>
#include <windows.h>
llama_model* load()
{
    llama_model_params ctx_params = llama_model_default_params();

    ctx_params.n_gpu_layers = 20;     // GPU加速层数（和cli保持一致）
    ctx_params.use_mmap = true;       // 启用内存映射，加快模型加载
    ctx_params.use_mlock = false;

    // 2. 加载千问模型
    const std::string model_path = R"(E:\ZeroPlan\python\dnn\models\qw\qwen1_5-0_5b-chat-q4_0.gguf)";
    llama_model* model = llama_load_model_from_file(model_path.c_str(), ctx_params);

    if (model == nullptr) {
        std::cerr << "load model failed" << std::endl;
        return nullptr;
    }
    std::cout << "model load success" << std::endl;

    return model;



}
#define MAX_TOKEN 10000

auto new_load()
{
    //初始化模型
    auto model_path = R"(E:\ZeroPlan\python\dnn\models\qw\qwen1_5-0_5b-chat-q4_0.gguf)";
    auto model_params = llama_model_default_params();
    auto model = llama_model_load_from_file(model_path, model_params);
    // 创建上下文
    struct llama_context_params context_params = llama_context_default_params();
    struct llama_context *context = llama_init_from_model(model, context_params);
    printf("create context down\n");

    // 获得词汇表
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    printf("create vocab down\n");

    // const char* prompt ="<|begin_of_text|><|start_header_id|>user<|end_header_id|> 介绍一下你自己 <|eot_id|><|start_header_id|>assistant<|end_header_id|>";
    const char* prompt =
        "<|im_start|>user\n"          // 用户角色开始
        "介绍一下你自己，你是什么类型的大语言模型\n"  // 提问（换行分隔，去掉多余空格）
        "<|im_end|>\n"                // 用户角色结束
        "<|im_start|>assistant\n";    // 助手角色开始（关键：结尾不加<|im_end|>，留给模型生成）
    // 对提示词进行标记化（tokenize）
    auto *tokens = static_cast<llama_token*>(malloc(sizeof(llama_token) * MAX_TOKEN));
    int len = llama_tokenize(
        vocab, prompt, strlen(prompt), tokens, MAX_TOKEN, false, true);
    if (len < 0){
        std::cerr << "Error：tokenize error" << std::endl;
        return -1;
    }
    printf("tokenize prompt down\n");

    // 创建批次
    struct llama_batch batch = llama_batch_get_one(tokens, len);
    printf("create batch down\n");

    // 初始化采样器链
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    struct llama_sampler *sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.8));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(50));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9, 1));
    long seed = time(nullptr);
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed));
    printf("create sampler chain down\n");

    // 循环
    llama_token next_token = LLAMA_TOKEN_NULL;
    llama_token eos = llama_vocab_eos(vocab);
    std::string  res;
    while (next_token != eos) {
        // 解码
        if(llama_decode(context, batch)){
            std::cerr << "Error: decode error" << std::endl;
            return -1;
        }

        // 采样
        next_token = llama_sampler_sample(sampler, context, -1);

        // 反标记化
        char deprompt[100] = {0};
        if(llama_detokenize(vocab, &next_token, 1, deprompt, sizeof(deprompt) / sizeof(deprompt[0]), false, false) < 0){
            std::cerr << "Error: detokenize error" << std::endl;
            return -1;
        }
        std::cout << deprompt;
        res += deprompt;
        // 更新 batch 以包含新生成的 token
        batch.token[0] = next_token;
        batch.n_tokens = 1;
    }
    // std::cout << "res: "<<res << std::endl;
    // 释放资源
    llama_sampler_free(sampler);
    llama_batch_free(batch);
    llama_free(context);
    llama_model_free(model);
    free(tokens);

    return 0;
}




int main()
{
    SetConsoleCP(CP_UTF8);       // 控制台输入编码
    SetConsoleOutputCP(CP_UTF8); // 控制台输出编码
    new_load();
    return 0;
}
