//
// Created by HP on 2025/12/22.
//
#include "llama.h"
#include <iostream>
#include <windows.h>
#include <nlohmann/json.hpp>

class LLmAI
{
    llama_model_params ctx_params = llama_model_default_params(); // 模型参数
    llama_model* model = nullptr; // 模型指针
    llama_context_params context_params = llama_context_default_params(); // 上下文参数
    llama_context* context = nullptr; // 上下文指针
    const llama_vocab* vocab = nullptr; // 词汇表指针
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler*  sampler = llama_sampler_chain_init(sparams); // 采样器指针

    public:
        explicit LLmAI(const std::string& model_path)
            {
                ctx_params.n_gpu_layers = 20;     // GPU加速层数（和cli保持一致）
                ctx_params.use_mmap = true;       // 启用内存映射，加快模型加载
                ctx_params.use_mlock = false;
                model = llama_model_load_from_file(model_path.c_str(), ctx_params);
                if (model == nullptr) {
                    std::cerr << "load model failed" << std::endl;
                    return;
                }
                std::cout << "model load success" << std::endl;
            }
            LLmAI& createContext()
            {

                context = llama_init_from_model(model, context_params);
                context_params.n_ctx = 2048;          // 上下文窗口大小（建议1024-4096，根据模型支持调整）
                context_params.n_batch = 512;         // 单次batch处理大小

                if (context == nullptr) {
                    std::cerr << "create context failed" << std::endl;
                    return *this;
                }
                std::cout << "context create success" << std::endl;
                return *this;
            }
            LLmAI& getVocab()
            {
                vocab = llama_model_get_vocab(model);
                if (vocab == nullptr) {
                    std::cerr << "get vocab failed" << std::endl;
                    return *this;
                }
                std::cout << "vocab get success" << std::endl;
                return *this;
            }

            LLmAI& createSParams()
            {
                llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.2));
                llama_sampler_chain_add(sampler, llama_sampler_init_top_k(30));
                llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9, 1));
                const long seed = time(nullptr);
                llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed));
                std::cout << "create sampler is success" << std::endl;
                return *this;
            }
            LLmAI& send(std::string sendstr,std::string& response)
            {
                std::string prompt =
               "<|im_start|>user\n"          // 用户角色开始
               + sendstr + "\n"             // 提问（换行分隔，去掉多余空格）
               "<|im_end|>\n"                // 用户角色结束
               "<|im_start|>assistant\n";    // 助手角色开始（关键：结尾不加<|im_end|>，留给模型生成）
                auto *tokens = static_cast<llama_token*>(malloc(sizeof(llama_token) * 10000));
                int len = llama_tokenize(
                        vocab, prompt.c_str(),
                        static_cast<int32_t>(prompt.size()),
                        tokens, 10000,
                        false, true
                        );
                if (len < 0){
                    std::cerr << "tokenize failed" << std::endl;
                    return *this;
                }
                llama_batch batch = llama_batch_get_one(tokens, len);
                llama_token next_token = LLAMA_TOKEN_NULL;
                llama_token eos = llama_vocab_eos(vocab);

                while (next_token != eos)
                {
                    if(llama_decode(context, batch)){
                        std::cerr << "Error: decode error" << std::endl;
                        return *this;
                    }
                    next_token = llama_sampler_sample(sampler, context, -1);
                    char deprompt[100] = {0};
                    if(llama_detokenize(vocab, &next_token, 1, deprompt, sizeof(deprompt) / sizeof(deprompt[0]), false, false) < 0){
                        std::cerr << "Error: detokenize error" << std::endl;
                        return *this;
                    }
                    std::cout << deprompt;
                    batch.token[0] = next_token;
                    batch.n_tokens = 1;
                    response += deprompt;
                }
                llama_batch_free(batch);
                return *this;
            }
            LLmAI& clearCache()
            {
                llama_model_kv_override(context);
                llama_sampler_reset(sampler);
                return *this;
            }
            ~LLmAI()
            {
                llama_sampler_free(sampler);
                llama_free(context);
                llama_model_free(model);
            }

};
std::string get_to_json(std::string& response)
{
    size_t start = response.find("```json");
    if (start != std::string::npos) {
        start += 7;  // 跳过```json
    } else {
        start = 0;
    }
    // 2. 找到结尾的```标记（从start位置开始找）
    size_t end = response.find("```", start);
    if (end == std::string::npos) {
        // 无结尾标记，取到最后一个非空白字符
        end = response.find_last_not_of(" \t\n\r") + 1;
    } else {
        // 有结尾标记，取到标记前的最后一个非空白字符
        end = response.find_last_not_of(" \t\n\r", end - 1) + 1;
    }
    return response.substr(start, end - start);
}
int str_for_int(std::string str)
{
    return std::stoi(str);
}

void func1() { std::cout << "func1" << std::endl; };
void func2() { std::cout << "func2" << std::endl; };
void func3() { std::cout << "func3" << std::endl; };
int add(int a,int b){std::cout << "add:" << a+b << std::endl; return a+b;}
void llama_log_callback_null(ggml_log_level level, const char * text, void * user_data)
{
    (void) level; (void) text; (void) user_data;
}


void  chat()
{
    SetConsoleCP(CP_UTF8);       // 控制台输入编码
    SetConsoleOutputCP(CP_UTF8); // 控制台输出编码
    auto model_path = R"(E:\ZeroPlan\python\dnn\models\qw\qwen1_5-0_5b-chat-q4_0.gguf)";
    llama_log_set(llama_log_callback_null, nullptr);

    LLmAI llm_ai(model_path);
    llm_ai.createContext().getVocab().createSParams();
    std::string response;
    while (true)
    {
        std::string int_put;
        std::cin>>int_put;
        llm_ai.createContext().send(int_put, response);
        std::cout << ""<<std::endl;
        if (int_put == "exit")
        {
            break;
        }
    }

}
void call_func()
{
    SetConsoleCP(CP_UTF8);       // 控制台输入编码
    SetConsoleOutputCP(CP_UTF8); // 控制台输出编码
    auto model_path = R"(E:\ZeroPlan\python\dnn\models\qw\qwen1_5-0_5b-chat-q4_0.gguf)";
    llama_log_set(llama_log_callback_null, nullptr);

    LLmAI llm_ai(model_path);
    llm_ai.createContext().getVocab().createSParams();
    std::string response;

    while (true)
    {
        std::string ci_int_put;
        std::cin>>ci_int_put;
        std::string int_put=R"(
        请严格遵守以下要求输出JSON，无任何多余内容（包括换行、空格、注释）：
            1. 函数列表及参数规则：
            - getWeather：参数为param1（城市名称，字符串类型），无其他参数
            - getUser：参数为param1（用户ID，字符串类型），无其他参数
            - add：参数为param1（整数）,param2（整数），无其他参数
            2. 其中严格按照函数列表中的参数规则，不能添加其他参数
            3. 输出格式：仅输出单个JSON对象，格式为{"funName":"函数名","param1":"参数值"...}，禁止添加多余参数、换行、空格
            4. 本次任务：)"+ci_int_put;
        std::cout << int_put<<std::endl;
        llm_ai.createContext().send(int_put, response);

    }


    auto json_str=get_to_json(response);
    nlohmann::json json = nlohmann::json::parse(json_str);
    std::cout<<json.dump(4)<<std::endl;
    // typedef int (*FuncPrt)(int,int);
    // std::map<std::string,FuncPrt> func_map={
    //    {"add",add},
    // };
    // auto ret = func_map[json["funName"].get<std::string>()](json["param1"].get<int>(), json["param2"].get<int>());
    // // std::cout << "ret:" << ret << std::endl;
    // auto result_info = "add(5,6)的计算结果是" + std::to_string(ret)+"告诉我结果是多少";
    // llm_ai.createContext().send(result_info, response);

}
int main()
{
    // SetConsoleCP(CP_UTF8);       // 控制台输入编码
    // SetConsoleOutputCP(CP_UTF8); // 控制台输出编码
    // auto model_path = R"(E:\ZeroPlan\python\dnn\models\qw\qwen1_5-0_5b-chat-q4_0.gguf)";
    // LLmAI llm_ai(model_path);
    // std::string response;
    // std::string int_put=R"(
    //              请遵循以下JSON格式，无任何多余内容：
    //             {"funName":"add","param1":int1,"param2":int2}
    //             现在调用add函数，参数为5和6，请直接输出上述格式的JSON，等到出结果后告诉我结果为多少，不要重复输出
    //     )";
    // std::string response2;
    // llm_ai.
    //     createContext().
    //     getVocab().
    //     createSParams().
    //     send(int_put, response);
    // // llm_ai.send("请写一段python代码，实现一个简单的加法函数", response2);
    // auto json_str=get_to_json(response);
    // nlohmann::json json = nlohmann::json::parse(json_str);
    // typedef int (*FuncPrt)(int,int);
    // std::map<std::string,FuncPrt> func_map={
    //    {"add",add},
    // };
    // auto ret = func_map[json["funName"].get<std::string>()](json["param1"].get<int>(), json["param2"].get<int>());
    //
    // // 手动将结果添加到对话历史（作为用户告知模型的信息）
    // std::string result_info = "add(5,6)的计算结果是" + std::to_string(ret);
    // llm_ai.send(result_info, response2);  // 先让模型“知道”结果
   // std::cout <<"r2:"<< response2 << std::endl;
    // chat();
     call_func();
    return 0;
}