//
// Created by HP on 2026/1/31.
//

#ifndef GENERALDNNLIB_ZERO_LLM_GGUF_STREAM_H
#define GENERALDNNLIB_ZERO_LLM_GGUF_STREAM_H
namespace gdlz::llm::data
{
    struct  LLM_GGUF_Stream {
        char stream[1024] = {0};
        int  str_len= 0;
    };

    struct LLM_GGUF_Embedding
    {
        std::unique_ptr<float[]> embeddings;
        int  embeddings_len= 0;

        void Reset()
        {
            embeddings.reset();
            embeddings_len = 0;
        }
    };
}
#endif //GENERALDNNLIB_ZERO_LLM_GGUF_STREAM_H
