//
// Created by HP on 2026/7/4.
//

#ifndef GENERALDNNLIB_ZERO_TRT_CV_OCR_EXPORT_H
#define GENERALDNNLIB_ZERO_TRT_CV_OCR_EXPORT_H
#include "trt_cv_ocr_framework.h"
#include "gdlz/export.h"
namespace gdlz::trt::icv::ocr {
    GDLZ_CORE_API TensorRTCvOCRFramework* CreateTensorRTCvOCRFramework(const char* config_path);
    GDLZ_CORE_API void DestroyTensorRTCvOCRFramework(TensorRTCvOCRFramework* framework);
    GDLZ_CORE_API int CreateCtx(TensorRTCvOCRFramework* framework,TensorRTCvOCRCtx& ctx);

}
#endif //GENERALDNNLIB_ZERO_TRT_CV_OCR_EXPORT_H