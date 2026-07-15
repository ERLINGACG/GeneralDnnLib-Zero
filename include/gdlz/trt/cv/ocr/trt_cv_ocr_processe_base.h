//
// Created by HP on 2026/7/4.
//

#ifndef GENERALDNNLIB_ZERO_TRT_CV_OCR_PROCESSER_BASE_H
#define GENERALDNNLIB_ZERO_TRT_CV_OCR_PROCESSER_BASE_H
#include <memory>
#include <opencv2/core/mat.hpp>

#include "gdlz/export.h"

namespace gdlz::trt::icv::ocr {

    struct PreParams{
        int32_t img_width;
        int32_t img_height;
        double  scalefactor;
    };

    struct PostParams{
        int32_t img_width{};
        int32_t img_height{};
        int32_t  PAD = 5;
        float   unclip_ratio{};
        float   thresh{};
    };

    struct ProImg{
        std::unique_ptr<cv::Mat> org_mat;
        std::unique_ptr<cv::Mat> blob_mat;
    };

    struct RectBoxs{
        std::unique_ptr<std::vector<cv::Rect>> rect_mat;

        auto getSize() const{
            if (!rect_mat){
                return 0;
            }
            return static_cast<int>(rect_mat->size());
        }
    };

}
#endif //GENERALDNNLIB_ZERO_TRT_CV_OCR_PROCESSER_BASE_H
