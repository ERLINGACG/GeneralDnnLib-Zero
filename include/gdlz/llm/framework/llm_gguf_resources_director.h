//
// Created by HP on 2026/1/26.
//

#ifndef GENERALDNNLIB_ZERO_LLM_GGUF_RESOURCES_H
#define GENERALDNNLIB_ZERO_LLM_GGUF_RESOURCES_H
#include "gdlz/res/llm_resources.h"
#include "gdlz/export.h"
#include <string>
namespace gdlz::resources::llm::gguf{
    struct LLm_GGuf_Resource :  LLm_GGuf_ResourceBase{
        std::string model_path;
    };

    class GDLZ_CPP_API LLm_GGuf_ResourceDirector : public LLm_GGuf_ResourceDirectorBase<LLm_GGuf_Resource,LLm_GGuf_ResourceDirector>{
    public:
        std::string conf_path;

        ~LLm_GGuf_ResourceDirector() override =default;
        LLm_GGuf_ResourceDirector& Hand(LLm_GGuf_Resource& resource) override;
        LLm_GGuf_ResourceDirector& SetConfPath(const std::string& conf_path);
    };
}
#endif //GENERALDNNLIB_ZERO_LLM_GGUF_RESOURCES_H