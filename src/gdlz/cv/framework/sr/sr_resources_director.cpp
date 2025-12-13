//
// Created by HP on 2025/12/11.
//
#include "gdlz/cv/framework/sr/sr_resources_director.h"

#include <fstream>
#include <dylog/logger.h>
#include <nlohmann/json.hpp>

using gdlz::resources::SrResourceDirector;


SrResourceDirector& SrResourceDirector::Hand(SrResource& resource)
{

        auto logger =dylog::DynamicLogger().setInvokeName(typeid(*this).name());

        auto load_conf=[&](nlohmann::json& j ,nlohmann::json& env)
        {
                env=j[j["env"].get<std::string>()];
                resource.type=env["type"].get<std::string>();
                resource.scale=env["scale"].get<int>();
                resource.model_path=env["model_path"].get<std::string>();
        };

        auto load_mode=[&]()
        {
                resource.sr.readModel(resource.model_path);
                resource.sr.setModel(resource.type, resource.scale);
        };
        try{
                const std::unique_ptr<std::ifstream> jsonfile(new std::ifstream(this->conf_path));
                nlohmann::json config;
                nlohmann::json env;
                * jsonfile >> config;
                load_conf(config,env);
                load_mode();
                logger.info("Loaded successfully");
        }
        catch (std::exception& e)
        {
                logger.error("load model failed: {}", e.what());
        }
        catch(...){
                logger.error("load model failed");
        }
        return *this;
}

SrResourceDirector& SrResourceDirector::SetConfPath(const std::string& conf_path)
{
        this->conf_path = conf_path;
        return *this;
}
