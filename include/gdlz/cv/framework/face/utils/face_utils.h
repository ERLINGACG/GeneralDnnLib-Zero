//
// Created by HP on 2026/1/14.
//

#ifndef GENERALDNNLIB_ZERO_FACE_UTILS_H
#define GENERALDNNLIB_ZERO_FACE_UTILS_H
#include "gdlz/export.h"
#include "gdlz/cv/framework/face/face_param.h"
#include "gdlz/cv/framework/face/base/feature_dnn.h"

namespace gdlz::cv_framework::face::utils
{
    GDLZ_CORE_API void GetFeatureVec512(
        const unsigned char* feature,
        int size
    );
    GDLZ_CORE_API double CompareVec512(
       unsigned char* feature1,
       unsigned char* feature2
    );
} // namespace gdlz::cv_framework::face::utils
#endif //GENERALDNNLIB_ZERO_FACE_UTILS_H