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
        resources::llm::gguf::LLm_GGuf_Resource resource{};
        std::string conf_path;

    public:
        explicit LLm_GGuf_Framework(std::string  config_path):conf_path(std::move(config_path)){};
        LLm_GGuf_Framework& ResourceDirector(resources::llm::gguf::LLm_GGuf_ResourceDirector& director);
        void releases();
        void ExampleSend() const;

        int32_t SendExample(const char* prompt,batch::LLM_GGUF_Batch& batch,data::LLM_GGUF_Stream& stream) const;
        void  InitBatch(const char* prompt,batch::LLM_GGUF_Batch& batch) const;
        void  Reasoning(batch::LLM_GGUF_Batch& batch,data::LLM_GGUF_Stream& stream) const;
    };
}
#endif //GENERALDNNLIB_ZERO_LLM_GGUF_FRAMEWORK_H