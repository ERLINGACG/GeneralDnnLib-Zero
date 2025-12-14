//
// Created by HP on 2026/1/7.
//

#ifndef GENERALDNNLIB_ZERO_YUNET_FRAMEWORK_EXPORT_H
#define GENERALDNNLIB_ZERO_YUNET_FRAMEWORK_EXPORT_H
#include "yunet_framework.h"
#include "gdlz/export.h"
#include "gdlz/cv/framework/face/arcface/arcface_framework.h"

namespace gdlz::cv_framework::face::yunet_export
{
    GDLZ_CORE_API yunet::CV_YuNetFramework* CreateYuNetFramework(const char* conf_path);
    GDLZ_CORE_API void DestroyYuNetFramework(const yunet::CV_YuNetFramework* framework);
    GDLZ_CORE_API void YuNetDetection       (yunet::CV_YuNetFramework* framework,
                                             int size, unsigned char* byte,
                                             data::YunetOutput& output
                                             );

    GDLZ_CORE_API void CV_YuNetDetection_FeatureVec_Arc(

        yunet::CV_YuNetFramework* framework,
        arcface::CV_ArcFaceFramework* arcface_framework,
        int size, unsigned char* byte,
        data::YunetOutput& output0,
        data::FeatureOutput& output1
    );

};


#endif //GENERALDNNLIB_ZERO_YUNET_FRAMEWORK_EXPORT_H