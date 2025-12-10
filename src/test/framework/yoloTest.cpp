//
// Created by HP on 2025/12/9.
//
#include <iostream>
#include "gdlz/cv/framework/yolo_framework.h"
using gdlz::resources::YoloResourceDirector;
int main()
{
    gdlz::cv_framework::YoloFramework framework("yolo.cfg");
    YoloResourceDirector director;
    framework.ResourceDirector(director);
    return 0;
}