//
// Created by HP on 2026/1/26.
//

#ifndef GENERALDNNLIB_ZERO_LLM_GGUF_FRAMEWORK_H
#define GENERALDNNLIB_ZERO_LLM_GGUF_FRAMEWORK_H
#include <string>

#include "llm_gguf_resources_director.h"
#include "gdlz/export.h"
namespace  gdlz::llm::gguf
{
    using  resources::llm::gguf::LLm_GGuf_Resource;
    using resources::llm::gguf::LLm_GGuf_ResourceDirector;

    class GDLZ_CPP_API LLm_GGuf_Framework {



        LLm_GGuf_Resource resource{};
        std::string conf_path;

    public:
        explicit LLm_GGuf_Framework(std::string  config_path):conf_path(std::move(config_path)){};
        LLm_GGuf_Framework& ResourceDirector(LLm_GGuf_ResourceDirector& director);
    };
}
#endif //GENERALDNNLIB_ZERO_LLM_GGUF_FRAMEWORK_H