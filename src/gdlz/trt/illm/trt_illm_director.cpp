//
// Created by HP on 2026/6/5.
//
#include "gdlz/trt/illm/trt_illm_director.h"
#include "gdlz/trt/core/trt_core_director.h"



int gdlz::trt::illm::TensorRTIllmDirector::Handle(TensorRTIllmEngine& engine, const char* config_path)
{
    core::TensorRTCoreDirector::Handle<TensorRTIllmEngine>(engine,config_path);
    return 0;
}
