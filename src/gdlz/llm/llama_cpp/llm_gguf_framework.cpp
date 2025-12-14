//
// Created by HP on 2026/1/27.
//
#include "gdlz//llm/framework/llm_gguf_framework.h"
using gdlz::llm::gguf::LLm_GGuf_Framework;

LLm_GGuf_Framework& LLm_GGuf_Framework::ResourceDirector(LLm_GGuf_ResourceDirector& director)
{
    director.SetConfPath(this->conf_path).Hand(this->resource);
    return *this;
}
