//
// Created by HP on 2026/1/13.
//
#include "gdlz/cv/framework/face/arcface/arcface_framework_export.h"
using gdlz::cv_framework::face::arcface::CV_ArcFaceFramework;
using gdlz::resources::CV_FeatureDnnResourceDirector;
CV_ArcFaceFramework* gdlz::cv_framework::face::arcface_export::CreateArcFaceFramework(const char* conf_path)
{
    auto framework = new CV_ArcFaceFramework();
    auto director = CV_FeatureDnnResourceDirector();
    framework->SetConfPath(conf_path).ResourceDirector(director);
    return framework;
}
void gdlz::cv_framework::face::arcface_export::DestroyArcFaceFramework(const CV_ArcFaceFramework* framework)
{
    delete framework;
}

