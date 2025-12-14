//
// Created by HP on 2026/1/6.
//

#ifndef GENERALDNNLIB_ZERO_FACE_PARAM_H
#define GENERALDNNLIB_ZERO_FACE_PARAM_H
#include <opencv2/core/types.hpp>

#include "gdlz/cv/framework/param.h"

namespace gdlz::cv_framework::param
{
    struct FaceParam : BaseParam{
        std::unique_ptr<cv::Mat> faces;
        std::unique_ptr<std::vector<cv::Rect>> rects = std::make_unique<std::vector<cv::Rect>>();
        // std::unique_ptr<>
    };
}
#endif //GENERALDNNLIB_ZERO_FACE_PARAM_H