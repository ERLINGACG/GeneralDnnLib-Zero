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

    struct LLM_GGUF_Context
    {
        llama_context* context = nullptr;
        llama_sampler* sampler = nullptr;

        ~LLM_GGUF_Context(){
            if (context != nullptr) {
                llama_free(context);
                context = nullptr;
            }
            if (sampler != nullptr) {
                llama_sampler_free(sampler);
                sampler = nullptr;
            }
        };
    };

    struct  LLM_GGUF_Embeddings_Ctx {

    };

    struct LLM_GGUF_Context_RTParam {

          int top_k{};
          float top_p{};
          float temp{};

          int n_ctx{};
          int n_batch{};

          char* gbnf_str{};
          bool  use_gbnf{};
          int  use_embeddings{};

          int     penalty_last_n{};   // last n tokens to penalize (0 = disable penalty, -1 = context size)
          float   penalty_repeat{};   // 1.0 = disabled
          float   penalty_freq{};     // 0.0 = disabled
          float   penalty_present{}; // 0.0 = disabled



    };
}
#endif //GENERALDNNLIB_ZERO_LLM_GGUF_BATCH_H