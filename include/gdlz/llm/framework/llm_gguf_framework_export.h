//
// Created by HP on 2026/1/27.
//

#ifndef GENERALDNNLIB_ZERO_LLM_GGUF_FRAMEWORK_EXPORT_H
#define GENERALDNNLIB_ZERO_LLM_GGUF_FRAMEWORK_EXPORT_H
#include "llm_gguf_framework.h"

namespace gdlz::llm::gguf_export
{
    GDLZ_CORE_API gguf::LLm_GGuf_Framework* CreateLLm_GGuf_Framework(const char* conf_path);
    GDLZ_CORE_API void                      DestroyLLm_GGuf_Framework(gguf::LLm_GGuf_Framework* framework);
    GDLZ_CORE_API void                      ExampleSend(const gguf::LLm_GGuf_Framework* framework);
}
#endif //GENERALDNNLIB_ZERO_LLM_GGUF_FRAMEWORK_EXPORT_H