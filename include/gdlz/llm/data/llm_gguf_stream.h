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
}
#endif //GENERALDNNLIB_ZERO_LLM_GGUF_STREAM_H
