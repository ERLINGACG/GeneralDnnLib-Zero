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
    GDLZ_CORE_API int32_t                   SendExample(
                                                const gguf::LLm_GGuf_Framework* framework,
                                                const char* prompt,
                                                gguf::batch::LLM_GGUF_Batch& batch,
                                                data::LLM_GGUF_Stream& stream
                                             );
    GDLZ_CORE_API void                     InitBatch(
                                                const gguf::LLm_GGuf_Framework* framework,
                                                const char* prompt,
                                                gguf::batch::LLM_GGUF_Batch& batch
                                             );
    GDLZ_CORE_API void                     Reasoning(
                                                const gguf::LLm_GGuf_Framework* framework,
                                                gguf::batch::LLM_GGUF_Batch& batch,
                                                data::LLM_GGUF_Stream& stream
                                             );
    GDLZ_CORE_API void                     Batch_Free(gguf::batch::LLM_GGUF_Batch& batch);
}
#endif //GENERALDNNLIB_ZERO_LLM_GGUF_FRAMEWORK_EXPORT_H