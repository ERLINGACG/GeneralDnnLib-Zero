//
// Created by HP on 2026/1/27.
//
#include "gdlz/llm/framework/llm_gguf_framework_export.h"

#include <iostream>
#include <ostream>
#include <nlohmann/json.hpp>

gdlz::llm::gguf::LLm_GGuf_Framework* gdlz::llm::gguf_export::CreateLLm_GGuf_Framework(const char* conf_path)
{
        std::cout << "CreateLLm_GGuf_Framework: " << conf_path << std::endl;
        const auto framework = new gguf::LLm_GGuf_Framework(conf_path);

        auto llm_gguf_resource_director = resources::llm::gguf::LLm_GGuf_ResourceDirector();
        // resources::llm::gguf::LLm_GGuf_Resource resource;
        // llm_gguf_resource_director.SetConfPath(conf_path);
        // llm_gguf_resource_director.Hand(resource);
        framework->ResourceDirector(llm_gguf_resource_director);
        return framework;
}

void gdlz::llm::gguf_export::DestroyLLm_GGuf_Framework(gguf::LLm_GGuf_Framework* framework)
{
        framework->releases();
        delete framework;

}
void gdlz::llm::gguf_export::SetSamplerRT(
        const gguf::LLm_GGuf_Framework* framework,
        const char* gbnf_str,
        const bool use_grammar,
        const int top_k,
        const float top_p,
        const float temp
        ) {

        framework->SetSamplerRT(gbnf_str, use_grammar, top_k, top_p, temp);
}

void gdlz::llm::gguf_export::InitBatch(const gguf::LLm_GGuf_Framework* framework, const char* prompt,
        gguf::batch::LLM_GGUF_Batch& batch) {
        framework->InitBatch(prompt, batch);
}
void gdlz::llm::gguf_export::Reasoning(const gguf::LLm_GGuf_Framework* framework, gguf::batch::LLM_GGUF_Batch& batch,
        data::LLM_GGUF_Stream& stream) {
        framework->Reasoning(batch, stream);
}


void gdlz::llm::gguf_export::Batch_Free(gguf::batch::LLM_GGUF_Batch& batch)
{
        llama_batch_free(*batch.batch);
}

void gdlz::llm::gguf_export::SetSamplerASync(
        const gguf::LLm_GGuf_Framework* framework,
        const gguf::batch::LLM_GGUF_Context_RTParam& param,
        gguf::batch::LLM_GGUF_Context& context
) {
        framework->SetSamplerASync(param, context);
}

int gdlz::llm::gguf_export::InitBatchASync(
        const gguf::LLm_GGuf_Framework* framework,
        const char* prompt,
        gguf::batch::LLM_GGUF_Batch& batch,
        gguf::batch::LLM_GGUF_Context& context
) {
        return framework->InitBatchASync(prompt, batch, context);
}

void gdlz::llm::gguf_export::ReasoningASync(
        const gguf::LLm_GGuf_Framework* framework,
        gguf::batch::LLM_GGUF_Batch& batch,
        gguf::batch::LLM_GGUF_Context& context,
        data::LLM_GGUF_Stream& stream
) {
        framework->ReasoningASync(batch, context, stream);
}

void gdlz::llm::gguf_export::InitEmbeddings(
        const gguf::LLm_GGuf_Framework* framework,
        const char* prompt,
        gguf::batch::LLM_GGUF_Batch& batch,
        gguf::batch::LLM_GGUF_Context& context
)
{
        framework->InitEmbeddings(prompt, batch, context);
}

void gdlz::llm::gguf_export::GetEmbeddings(
        const gdlz::llm::gguf::LLm_GGuf_Framework* framework,
        gdlz::llm::gguf::batch::LLM_GGUF_Batch& batch,
        gdlz::llm::gguf::batch::LLM_GGUF_Context& context)
{
        framework->GetEmbeddings(batch, context);
}



void gdlz::llm::gguf_export::Json_ExampleSend(const char* str)
{
        try {
                std::cout << str<< std::endl;
                auto json = nlohmann::json::parse(str);
                std::cout << json.dump(4) << std::endl;

        }catch (std::exception& e) {
                std::cerr << e.what() << std::endl;
        }

}
