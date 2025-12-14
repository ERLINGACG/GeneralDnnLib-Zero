//
// Created by HP on 2025/12/9.
//
#include <iostream>
#include "gdlz/cv/framework/yolo/yolo_framework.h"
using gdlz::resources::CV_YoloResourceDirector;
int main()
{
    gdlz::cv_framework::yolo::CV_YoloFramework framework("yolo.cfg");
    CV_YoloResourceDirector director;
    framework.ResourceDirector(director);
    return 0;
}