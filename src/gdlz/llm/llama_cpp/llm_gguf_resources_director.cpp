//
// Created by HP on 2026/1/26.
//
#include "gdlz/llm/framework/llm_gguf_resources_director.h"
#include "gdlz/llm/framework/llm_gguf_resources_director.h"
#include <llama.h>
#include <fstream>
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

        auto load_model=[&]()
        {
                resource.model_params->n_gpu_layers=20;
                resource.model_params->use_mmap=true;
                resource.model_params->use_mlock=false;

                resource.model=llama_model_load_from_file(
                                        resource.model_path.c_str(),
                                        *resource.model_params
                                );
                // logger.debug("use model_path: {}",resource.model_path.c_str());
                if (resource.model == nullptr) {
                        logger.error("load model failed");
                }else {
                        logger.debug("model load success");
                }
        };

        try{
                const unique_ptr<ifstream> jsonfile(new ifstream(this->conf_path));
                json config;
                json env;
                * jsonfile >> config;


                set_env(config,env);
                set_mode_path(env);
                load_model();
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

