//
// Created by HP on 2026/1/13.
//
#ifndef ARCFACE_FRAMEWORK_EXPORT_H
#define ARCFACE_FRAMEWORK_EXPORT_H


#include "arcface_framework.h"
#include "gdlz/export.h"
namespace gdlz::cv_framework::face::arcface_export
{
    GDLZ_CORE_API arcface::CV_ArcFaceFramework* CreateArcFaceFramework(const char* conf_path);
    GDLZ_CORE_API void DestroyArcFaceFramework(const arcface::CV_ArcFaceFramework* framework);


}
#endif
