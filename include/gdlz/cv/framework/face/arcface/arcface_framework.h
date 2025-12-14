//
// Created by HP on 2026/1/13.
//
#ifndef ARC_FACE_FRAMEWORK_EXPORT_H
#define ARCFACE_FRAMEWORK_EXPORT_H
#include "gdlz/cv/framework/face/base/feature_dnn.h"
#include "gdlz/export.h"
#include "gdlz/cv/data/feature_output.h"
#include "gdlz/cv/framework/face/face_param.h"

namespace gdlz::cv_framework::face::arcface
{
    class GDLZ_CPP_API CV_ArcFaceFramework : public base::CV_FeatureDnnBase<CV_ArcFaceFramework>
    {
        std::vector<cv::Point2f> standardLandmarks = {
            cv::Point2f(38.2946f, 51.6963f),   // 右眼
            cv::Point2f(73.5318f, 51.5014f),   // 左眼
            cv::Point2f(56.0252f, 71.7366f),   // 鼻子
            cv::Point2f(41.5493f, 92.3655f),   // 右嘴角
            cv::Point2f(70.7299f, 92.2041f)    // 左嘴角
        };
       public:
            CV_ArcFaceFramework& Process(param::FaceParam& param, data::FeatureOutput& feature_output);


    };
} // namespace gdlz::cv::framework::face::arcface
#endif
