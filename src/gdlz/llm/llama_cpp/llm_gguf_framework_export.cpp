//
// Created by HP on 2026/1/27.
//
#include "gdlz/llm/framework/llm_gguf_framework_export.h"

#include <iostream>
#include <ostream>

gdlz::llm::gguf::LLm_GGuf_Framework* gdlz::llm::gguf_export::CreateLLm_GGuf_Framework(const char* conf_path)
{
        const auto framework = new gguf::LLm_GGuf_Framework(conf_path);
        auto llm_gguf_resource_director = resources::llm::gguf::LLm_GGuf_ResourceDirector();
        framework->ResourceDirector(llm_gguf_resource_director);
        return framework;
}

void gdlz::llm::gguf_export::DestroyLLm_GGuf_Framework(gguf::LLm_GGuf_Framework* framework)
{
        framework->releases();
        delete framework;

}


void gdlz::llm::gguf_export::ExampleSend(const gguf::LLm_GGuf_Framework* framework)
{
        framework->ExampleSend();
}

int32_t gdlz::llm::gguf_export::SendExample(
        const gguf::LLm_GGuf_Framework* framework,
        const char* prompt,
        gguf::batch::LLM_GGUF_Batch& batch,
        data::LLM_GGUF_Stream& stream)
{
        try{
                return framework->SendExample(prompt, batch, stream);
                // return 0;
        }catch (std::exception& e) {
                std::cerr << e.what() << std::endl;
                return -1;
        }
}

void gdlz::llm::gguf_export::InitBatch(const gguf::LLm_GGuf_Framework* framework, const char* prompt,
        gguf::batch::LLM_GGUF_Batch& batch)
{
        framework->InitBatch(prompt, batch);
}
void gdlz::llm::gguf_export::Reasoning(const gguf::LLm_GGuf_Framework* framework, gguf::batch::LLM_GGUF_Batch& batch,
        data::LLM_GGUF_Stream& stream)
{
        framework->Reasoning(batch, stream);
}


void gdlz::llm::gguf_export::Batch_Free(gguf::batch::LLM_GGUF_Batch& batch)
{
        llama_batch_free(*batch.batch);
}

