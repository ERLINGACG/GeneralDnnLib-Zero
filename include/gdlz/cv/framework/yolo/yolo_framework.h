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



    class GDLZ_CPP_API YoloFramework {

            YoloResource resource{};
            std::string  conf_path;

        public:
            explicit YoloFramework(std::string  conf_path):conf_path(std::move(conf_path)){};
            ~YoloFramework() =default;

            YoloFramework& ResourceDirector(resources::YoloResourceDirector& director);
            YoloFramework& Input(cv::Mat& input,              param::YoloParam& param);
            YoloFramework& Input(int size,unsigned char* byte,param::YoloParam& param);
            YoloFramework& SetBlob(param::YoloParam& param);
            YoloFramework& Forward(param::YoloParam& param);
            YoloFramework& Process(param::YoloParam& param);
            YoloFramework& PostProcess(param::YoloParam& param);
            YoloFramework& GetOutput(data::YoloOutput& output, const param::YoloParam& param);




    };
}
#endif //YOLO_FRAMEWORK_H
