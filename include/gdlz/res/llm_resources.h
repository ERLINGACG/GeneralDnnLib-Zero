//
// Created by HP on 2026/1/26.
//

#ifndef GENERALDNNLIB_ZERO_LLM_RESOURCES_H
#define GENERALDNNLIB_ZERO_LLM_RESOURCES_H
#include <llama.h>
#include <memory>



namespace gdlz::resources::llm{

    struct LLm_GGuf_ResourceBase {
        llama_model_params   model_params  =llama_model_default_params();
        llama_context_params context_params=llama_context_default_params();
        llama_sampler_chain_params sampler_chain_params=llama_sampler_chain_default_params();
        const llama_vocab* vocab = nullptr;
        llama_context* context = nullptr;
        llama_sampler* sampler =  llama_sampler_chain_init(sampler_chain_params);
        llama_model* model{};

        void release(){
            if (context != nullptr) {
                llama_free(context);
                context = nullptr;
            }
            if (model != nullptr) {
                llama_model_free(model);
                model = nullptr;
            }
            if (sampler != nullptr) {
                llama_sampler_free(sampler);
                sampler = nullptr;
            }
        }
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