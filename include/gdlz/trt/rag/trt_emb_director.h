//
// Created by HP on 2026/4/19.
//

#ifndef GENERALDNNLIB_ZERO_TRT_EMB_DIRECTOR_H
#define GENERALDNNLIB_ZERO_TRT_EMB_DIRECTOR_H
#include "trt_emb_data.h"
// #include <nlohmann/json>
namespace gdlz::trt::rag
{
    class TrtEmbDirector {
        public:
            TrtEmbDirector() = default;
            ~TrtEmbDirector() = default;
            TrtEmbDirector& Handle(TrtEmbEngine& engine,const char* config_Path);

    };
}
#endif //GENERALDNNLIB_ZERO_TRT_EMB_DIRECTOR_H