//
// Created by HP on 2025/12/10.
//

#ifndef YOLO_PARAM_H
#define YOLO_PARAM_H
#include <string>
#include <opencv2/core/mat.hpp>
#include <vector>

#include "../param.h"

namespace gdlz::cv_framework::param
{

    struct YoloParam : BaseParam
    {

        std::unique_ptr<cv::Mat> forward_output;
        std::unique_ptr<std::vector<cv::Rect>> boxes       =std::make_unique<std::vector<cv::Rect>>();
        std::unique_ptr<std::vector<float>> confidences    =std::make_unique<std::vector<float>>();
        std::unique_ptr<std::vector<int>> class_ids        =std::make_unique<std::vector<int>>();
        std::unique_ptr<std::vector<int>> indices          =std::make_unique<std::vector<int>>();
        std::unique_ptr<std::string> describe_str          =std::make_unique<std::string>();
    };
}
#endif //YOLO_PARAM_H
