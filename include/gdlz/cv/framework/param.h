//
// Created by HP on 2025/12/11.
//

#ifndef PARAM_H
#define PARAM_H
#include <memory>

namespace gdlz::cv_framework::param
{
    struct BaseParam
    {

        std::unique_ptr<cv::Mat> input;
        std::unique_ptr<cv::Mat> blob;
        virtual ~BaseParam() = default;
    };
}
#endif //PARAM_H
