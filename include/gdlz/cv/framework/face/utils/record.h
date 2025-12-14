//
// Created by HP on 2026/1/14.
//

#ifndef GENERALDNNLIB_ZERO_RECORD_H
#define GENERALDNNLIB_ZERO_RECORD_H
#include "gdlz/cv/framework/face/face_param.h"
#include "gdlz/cv/framework/face/base/feature_dnn.h"

namespace gdlz::cv_framework::utils
{
    template<typename T, typename F>
    struct FaceRecord {

        std::unique_ptr<T> Face_Detector;
        std::unique_ptr<F> Face_Feature_Extractor;

       void Clear() {

           Face_Detector.reset();
           Face_Feature_Extractor.reset();
       }
    };
}
#endif //GENERALDNNLIB_ZERO_RECORD_H