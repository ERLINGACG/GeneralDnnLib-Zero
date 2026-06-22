//
// Created by HP on 2026/6/5.
//

#ifndef GENERALDNNLIB_ZERO_TRT_ILLM_DIRECTOR_H
#define GENERALDNNLIB_ZERO_TRT_ILLM_DIRECTOR_H
#include "gdlz/export.h"
#include "trt_illm_data.h"

namespace gdlz::trt::illm{
    class GDLZ_CPP_API TensorRTIllmDirector {
        public:
             TensorRTIllmDirector()=default;
             ~TensorRTIllmDirector()=default;

             static int Handle(TensorRTIllmEngine& engine,const char* config_path);
    };
}
#endif //GENERALDNNLIB_ZERO_TRT_ILLM_DIRECTOR_H