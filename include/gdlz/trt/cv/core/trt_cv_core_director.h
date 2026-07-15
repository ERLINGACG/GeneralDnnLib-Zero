//
// Created by HP on 2026/7/8.
//

#ifndef GENERALDNNLIB_ZERO_TRT_CV_CORE_DIRECTOR_H
#define GENERALDNNLIB_ZERO_TRT_CV_CORE_DIRECTOR_H
#include "trt_cv_core_data.h"
#include "gdlz/export.h"

namespace gdlz::trt::icv::core{
    class GDLZ_CPP_API TensorRTCvCoreDirector{
    public:
        static int Handle(TensorRTCvEngine& engine,const char* config_path);
    };
}
#endif //GENERALDNNLIB_ZERO_TRT_CV_CORE_DIRECTOR_H