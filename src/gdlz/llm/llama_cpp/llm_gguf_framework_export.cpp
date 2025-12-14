//
// Created by HP on 2026/1/27.
//
#include "gdlz/llm/framework/llm_gguf_framework_export.h"

gdlz::llm::gguf::LLm_GGuf_Framework* gdlz::llm::gguf_export::CreateLLm_GGuf_Framework(const char* conf_path)
{
        const auto framework = new gguf::LLm_GGuf_Framework(conf_path);
        auto llm_gguf_resource_director = gguf::LLm_GGuf_ResourceDirector();
        framework->ResourceDirector(llm_gguf_resource_director);
        return framework;
}

void gdlz::llm::gguf_export::DestroyLLm_GGuf_Framework(const gguf::LLm_GGuf_Framework* framework)
{
        delete framework;
}
