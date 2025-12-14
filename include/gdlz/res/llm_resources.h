//
// Created by HP on 2026/1/26.
//

#ifndef GENERALDNNLIB_ZERO_LLM_RESOURCES_H
#define GENERALDNNLIB_ZERO_LLM_RESOURCES_H
#include <llama.h>
#include <memory>

namespace gdlz::resources::llm{

    struct LLm_GGuf_ResourceBase {
        std::unique_ptr<llama_model_params>   model_params  =std::make_unique<llama_model_params>();
        std::unique_ptr<llama_context_params> context_params=std::make_unique<llama_context_params>();
        llama_model* model;

       // virtual  ~LLm_GGuf_ResourceBase()=default;
    };

    template<typename  T,typename R>
    class LLm_GGuf_ResourceDirectorBase
    {
    public:

        virtual ~LLm_GGuf_ResourceDirectorBase()=default;
        virtual R& Hand(T& resource)=0;
    };
}
#endif //GENERALDNNLIB_ZERO_LLM_RESOURCES_H