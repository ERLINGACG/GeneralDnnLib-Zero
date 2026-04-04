//
// Created by HP on 2026/1/26.
//
#include "gdlz/llm/framework/llm_gguf_resources_director.h"
#include <llama.h>
#include <fstream>
#include <iostream>
#include <dylog/logger.h>

#include <nlohmann/json.hpp>
using gdlz::resources::llm::gguf::LLm_GGuf_ResourceDirector;
using nlohmann::json;
LLm_GGuf_ResourceDirector& LLm_GGuf_ResourceDirector::Hand(LLm_GGuf_Resource& resource)
{
        using std::unique_ptr;
        using std::ifstream;

        auto logger =dylog::DynamicLogger().setInvokeName("LLM_GGUF_ResourceDirector");


        auto set_env=[&](json j,json& env)
        {
                env=j[j["env"].get<std::string>()];
                logger.debug("use env: {}",j["env"].get<std::string>().c_str());

        };
        auto set_mode_path=[&](json env)
        {
                resource.model_path = env["model_path"].get<std::string>();
                logger.debug("use model_path: {}",resource.model_path.c_str());
        };

        // auto set_model_params=[&](json env)
        // {
        //         resource.model_params.n_gpu_layers=env["n_gpu_layers"].get<int>();
        // };

        auto load_model=[&](json env)
        {

                resource.model_params.n_gpu_layers=env["n_gpu_layers"].get<int>();
                resource.model_params.use_mmap=env["use_mmap"].get<bool>();
                resource.model_params.use_mlock=env["use_mlock"].get<bool>();
                ggml_backend_dev_t gpu1_dev = ggml_backend_dev_get(0);
                resource.model_params.devices = new ggml_backend_dev_t[2]{gpu1_dev, nullptr};
                resource.model=llama_model_load_from_file(
                                        resource.model_path.c_str(),
                                        resource.model_params
                                );
                // logger.debug("use model_path: {}",resource.model_path.c_str());
                if (resource.model == nullptr) {
                        logger.error("load model failed");
                }else {
                        logger.debug("model load success");
                }

        };
        // auto set_context=[&](json env)
        // {
        //         resource.context_params.n_ctx=env["n_ctx"].get<int>();
        //         resource.context_params.n_batch=env["n_batch"].get<int>();
        //         // struct llama_sampler_seq_config samplers={
        //         //         0,
        //         //         resource.sampler
        //         // };
        //        // resource.context_params.samplers =&samplers;
        //         resource.context=llama_init_from_model(resource.model, resource.context_params);
        //
        //         if (resource.context == nullptr) {
        //                 logger.error("init context failed");
        //         }else {
        //                 logger.debug("context init success");
        //         }
        // };
        auto set_llama_vocab=[&]()
        {
                resource.vocab=llama_model_get_vocab(resource.model);
                if (resource.vocab == nullptr) {
                        logger.error("init vocab failed");
                }else {
                        logger.debug("vocab init success");
                }
        };

        // auto set_sampler=[&](json env)
        // {
        //         if (env["is_grammar"].get<bool>()){
        //                 // const char *grammar_str = R"(
        //                 // root ::="answer:" "{" name "," age "}"
        //                 // name ::=" 'name' :"[^,]+
        //                 // age  ::=" 'age' :"[0-9]+
        //                 // )";
        //                 std::string grammar_str = env["grammar_file"].get<std::string>();
        //                 logger.setInvokeName("test_grammar_sampler").debug(grammar_str.c_str());
        //
        //                 std::ifstream grammar_file(grammar_str);
        //                 if (!grammar_file.is_open()) {
        //                         logger.error("init grammar file failed");
        //                         return;
        //                 }
        //                 std::stringstream grammar_file_stream;
        //                 grammar_file_stream << grammar_file.rdbuf();
        //
        //                 struct llama_sampler* grammar_sampler = llama_sampler_init_grammar(
        //                   resource.vocab,          // 模型词汇表
        //                   grammar_file_stream.str().c_str(),// GBNF规则字符串
        //                        "root"          // 根规则名
        //                    );
        //                 if (grammar_sampler == nullptr) {
        //                         logger.error("init grammar sampler failed");
        //                 }else{
        //                         llama_sampler_chain_add(resource.sampler, grammar_sampler);
        //                 }
        //         }else {
        //               logger.debug("grammar sampler not use");
        //         }
        //         llama_sampler_chain_add(resource.sampler, llama_sampler_init_temp(env["temp"].get<float>()));
        //         llama_sampler_chain_add(resource.sampler, llama_sampler_init_top_k(env["top_k"].get<int>()));
        //         llama_sampler_chain_add(resource.sampler, llama_sampler_init_top_p(env["top_p"].get<float>(), 1));
        //         llama_sampler_chain_add(resource.sampler, llama_sampler_init_dist(time(nullptr)));
        //
        // };

        try{
                const unique_ptr<ifstream> jsonfile(new ifstream(this->conf_path));
                json config;
                json env;
                * jsonfile >> config;


                set_env(config,env);
                set_mode_path(env);
                load_model(env);
                // set_sampler(env);
                // set_context(env);
                set_llama_vocab();

        }catch (std::exception& e) {
                logger.error("exception: {}",e.what());
        }
        resource.model_path = conf_path;
        return *this;
}

LLm_GGuf_ResourceDirector& LLm_GGuf_ResourceDirector::SetConfPath(const std::string& conf_path)
{
        this->conf_path = conf_path;
        return *this;
}

