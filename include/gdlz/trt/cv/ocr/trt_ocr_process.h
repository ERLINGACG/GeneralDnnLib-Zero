//
// Created by HP on 2026/7/4.
//

#ifndef GENERALDNNLIB_ZERO_TRT_OCR_PROCESSER_H
#define GENERALDNNLIB_ZERO_TRT_OCR_PROCESSER_H
#include <opencv2/core/mat.hpp>

#include "trt_cv_ocr_processe_base.h"
#include "gdlz/export.h"
#include "gdlz/trt/core/cuda/trt_cuda_data.h"

namespace gdlz::trt::icv::ocr{

       class GDLZ_CPP_API TensorRTCvOCRProcesserDet{
           public:
               int MatToProImg(cv::Mat& img, ProImg& pro_img);
               int PreProcess (ProImg& img, PreParams params);
               int PostProcess(float* output,ProImg& img, PostParams params, RectBoxs& rect_boxs);
       };


       class GDLZ_CPP_API TensorRTCvOCRProcesserReC {
           public:
               int MatToProImg(cv::Mat& img, ProImg& pro_img);
               int PreProcess (ProImg& img, PreParams params);
               int PostProcess(float* output,ProImg& img, PostParams params, RectBoxs& rect_boxs);
       };
}
#endif //GENERALDNNLIB_ZERO_TRT_OCR_PROCESSER_H
