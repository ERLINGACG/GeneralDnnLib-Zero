//
// Created by HP on 2026/7/2.
//

#ifndef GENERALDNNLIB_ZERO_TRT_CV_OCR_DIRECTOR_H
#define GENERALDNNLIB_ZERO_TRT_CV_OCR_DIRECTOR_H
#include "trt_cv_ocr_data.h"

namespace gdlz::trt::icv::ocr {

    class TensorRTCvOCRDirector{
    public:
        static int Handle(TensorRTCvOCREngine& engine,const char* config_path);
    };
}
#endif //GENERALDNNLIB_ZERO_TRT_CV_OCR_DIRECTOR_H