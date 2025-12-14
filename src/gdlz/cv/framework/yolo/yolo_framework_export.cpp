//
// Created by HP on 2025/12/14.
//
#include "gdlz/cv/framework/yolo/yolo_framework_export.h"

using gdlz::cv_framework::yolo::CV_YoloFramework;

CV_YoloFramework* gdlz::cv_framework::yolo_export::CreateYoloFramework(const char* conf_path)
{
    const auto framework=new CV_YoloFramework(conf_path);
    auto director=resources::CV_YoloResourceDirector();
    framework->ResourceDirector(director);
    return framework;
}

void gdlz::cv_framework::yolo_export::DestroyYoloFramework(const CV_YoloFramework* framework)
{
    delete framework;
}

void gdlz::cv_framework::yolo_export::YoloDetection(CV_YoloFramework* framework,
    int size, unsigned char* byte,
    data::YoloOutput& output
){
    auto param=param::YoloParam();
    framework->Input(size,byte,param)
        .SetBlob(param)
        .Forward(param)
        .Process(param)
        .PostProcess(param)
        .GetOutput(output,param);
}


