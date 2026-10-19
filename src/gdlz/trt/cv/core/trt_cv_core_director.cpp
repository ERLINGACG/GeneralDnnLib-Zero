//
// Created by HP on 2026/7/8.
//
#include "gdlz/trt/cv/core/trt_cv_core_director.h"

#include "gdlz/trt/core/trt_core_director.h"
#include "gdlz/trt/cv/core/private/trt_cv_core_impi.h"
using gdlz::trt::icv::core::TensorRTCvCoreDirector;
using gdlz::trt::core::TensorRTCoreDirector;
using gdlz::trt::icv::core::TensorRTCvEngineImpi;
int TensorRTCvCoreDirector::Handle(TensorRTCvEngine& engine, const char* config_path){
    const auto& engine_impi=engine.impi();
    return TensorRTCoreDirector::Handle(*engine_impi,config_path);
}
