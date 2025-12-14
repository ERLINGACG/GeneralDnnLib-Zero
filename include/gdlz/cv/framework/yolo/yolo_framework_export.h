//
// Created by HP on 2025/12/14.
//

#ifndef YOLO_FRAMEWORK_EXPORT_H
#define YOLO_FRAMEWORK_EXPORT_H
#include "yolo_framework.h"
#include "gdlz/export.h"
namespace gdlz::cv_framework::yolo_export
{
    GDLZ_CORE_API yolo::CV_YoloFramework* CreateYoloFramework(const char* conf_path);
    GDLZ_CORE_API void                 DestroyYoloFramework(const yolo::CV_YoloFramework* framework);
    GDLZ_CORE_API void                 YoloDetection(yolo::CV_YoloFramework* framework,
                                                     int size,unsigned char* byte,
                                                     data::YoloOutput& output);

}
#endif //YOLO_FRAMEWORK_EXPORT_H
