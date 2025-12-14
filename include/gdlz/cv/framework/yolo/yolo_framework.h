//
// Created by HP on 2025/12/9.
//

#ifndef YOLO_FRAMEWORK_H
#define YOLO_FRAMEWORK_H

#include <utility>

#include "yolo_param.h"
#include "gdlz/export.h"
#include "gdlz/cv/data/out_put.h"
#include "yolo_resources.h"

namespace gdlz::cv_framework::yolo
{
    using resources::YoloResource;



    class GDLZ_CPP_API CV_YoloFramework {

            YoloResource resource{};
            std::string  conf_path;

        public:
            explicit CV_YoloFramework(std::string  conf_path):conf_path(std::move(conf_path)){};
            ~CV_YoloFramework() =default;

            CV_YoloFramework& ResourceDirector(resources::CV_YoloResourceDirector& director);
            CV_YoloFramework& Input(cv::Mat& input,              param::YoloParam& param);
            CV_YoloFramework& Input(int size,unsigned char* byte,param::YoloParam& param);
            CV_YoloFramework& SetBlob(param::YoloParam& param);
            CV_YoloFramework& Forward(param::YoloParam& param);
            CV_YoloFramework& Process(param::YoloParam& param);
            CV_YoloFramework& PostProcess(param::YoloParam& param);
            CV_YoloFramework& GetOutput(data::YoloOutput& output, const param::YoloParam& param);




    };
}
#endif //YOLO_FRAMEWORK_H
