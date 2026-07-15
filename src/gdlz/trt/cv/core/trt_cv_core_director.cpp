//
// Created by HP on 2026/7/8.
//
#include "gdlz/trt/cv/core/trt_cv_core_director.h"

#include "gdlz/trt/core/trt_core_director.h"

using gdlz::trt::icv::core::TensorRTCvCoreDirector;
using gdlz::trt::core::TensorRTCoreDirector;

int TensorRTCvCoreDirector::Handle(TensorRTCvEngine& engine, const char* config_path){
    return TensorRTCoreDirector::Handle(engine,config_path);
}
