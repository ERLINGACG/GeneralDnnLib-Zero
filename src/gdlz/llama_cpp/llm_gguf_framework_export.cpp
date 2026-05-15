//
// Created by HP on 2026/1/27.
//
#include "gdlz/llama/framework/llm_gguf_framework_export.h"

#include <iostream>
#include <ostream>
#include <nlohmann/json.hpp>

gdlz::llm::gguf::LLm_GGuf_Framework* gdlz::llm::gguf_export::CreateLLm_GGuf_Framework(const char* conf_path)
{
        std::cout << "CreateLLm_GGuf_Framework: " << conf_path << std::endl;
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


void gdlz::llm::gguf_export::Batch_Free(gguf::batch::LLM_GGUF_Batch& batch)
{
        llama_batch_free(*batch.batch);
}

void gdlz::llm::gguf_export::Embedding_Free(data::LLM_GGUF_Embedding& embedding)
{
        embedding.Reset();
}

void gdlz::llm::gguf_export::Context_Free(gguf::batch::LLM_GGUF_Context& context)
{
        context.~LLM_GGUF_Context();
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
        const gguf::LLm_GGuf_Framework* framework,
        gguf::batch::LLM_GGUF_Batch& batch,
        gguf::batch::LLM_GGUF_Context& context,
        data::LLM_GGUF_Embedding& embedding
)
{
        framework->GetEmbeddings(batch, context, embedding);
}


