//
// Created by HP on 2026/1/31.
//

#ifndef GENERALDNNLIB_ZERO_LLM_GGUF_BATCH_H
#define GENERALDNNLIB_ZERO_LLM_GGUF_BATCH_H
#include <llama.h>

namespace gdlz::llm::gguf::batch
{
    struct LLM_GGUF_Batch {
        std::unique_ptr<llama_batch> batch{};
        int next_token = LLAMA_TOKEN_NULL;
        int eos{};

    };
}
#endif //GENERALDNNLIB_ZERO_LLM_GGUF_BATCH_H