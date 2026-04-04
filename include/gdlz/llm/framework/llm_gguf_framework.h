//
// Created by HP on 2026/1/26.
//

#ifndef GENERALDNNLIB_ZERO_LLM_GGUF_FRAMEWORK_H
#define GENERALDNNLIB_ZERO_LLM_GGUF_FRAMEWORK_H
#include <string>

#include "llm_gguf_resources_director.h"
#include "gdlz/export.h"
#include "gdlz/llm/data/llm_gguf_batch.h"
#include "gdlz/llm/data/llm_gguf_stream.h"

namespace  gdlz::llm::gguf
{


    class GDLZ_CPP_API LLm_GGuf_Framework {
        resources::llm::gguf::LLm_GGuf_Resource resource;
        std::string conf_path;

    public:
        explicit LLm_GGuf_Framework(const char* config_path)
        {
                conf_path =config_path;
        };
        LLm_GGuf_Framework& ResourceDirector(resources::llm::gguf::LLm_GGuf_ResourceDirector& director);
        void releases();
        void ExampleSend() const;

        int32_t SendExample(const char* prompt,batch::LLM_GGUF_Batch& batch,data::LLM_GGUF_Stream& stream) const;

        void  SetSamplerRT(
            const char* gbnf_str,
            bool use_grammar,
            int top_k,
            float top_p,
            float temp
        ) const;
        void  InitBatch(const char* prompt,batch::LLM_GGUF_Batch& batch) const;
        void  Reasoning(batch::LLM_GGUF_Batch& batch,data::LLM_GGUF_Stream& stream) const;

        void  SetSamplerASync(
            const batch::LLM_GGUF_Context_RTParam& param,
            batch::LLM_GGUF_Context& context
        ) const;

        int  InitBatchASync(
            const char* prompt,
            batch::LLM_GGUF_Batch& batch,
            batch::LLM_GGUF_Context& context
        ) const;

        void ReasoningASync(
            batch::LLM_GGUF_Batch& batch,
            batch::LLM_GGUF_Context& context,
            data::LLM_GGUF_Stream& stream
        ) const;

        void InitEmbeddings(
             const char* prompt,
             batch::LLM_GGUF_Batch& batch,
             batch::LLM_GGUF_Context& context
            ) const;

        void InitEmbeddings_0(
            const char* prompt,
            batch::LLM_GGUF_Batch& batch,
            batch::LLM_GGUF_Context& context
            ) const;

        float* GetEmbeddings(
            batch::LLM_GGUF_Batch& batch,
            batch::LLM_GGUF_Context& context
            ) const;


    };
}
#endif //GENERALDNNLIB_ZERO_LLM_GGUF_FRAMEWORK_H