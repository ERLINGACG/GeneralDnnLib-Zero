//
// Created by HP on 2026/7/2.
//
#include "gdlz/trt/cv/ocr/trt_cv_ocr_director.h"

#include "gdlz/trt/core/trt_core_director.h"
using gdlz::trt::icv::ocr::TensorRTCvOCRDirector;

int TensorRTCvOCRDirector::Handle(TensorRTCvOCREngine& engine,const char* config_path) {
    return core::TensorRTCoreDirector::Handle(engine,config_path);
}