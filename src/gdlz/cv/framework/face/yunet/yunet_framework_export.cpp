//
// Created by HP on 2026/1/7.
//
#include "gdlz/cv/framework/face/yunet/yunet_framework_export.h"
#include "gdlz/cv/framework/face/yunet/yunet_resources.h"

using gdlz::cv_framework::face::yunet::CV_YuNetFramework;
using gdlz::resources::CV_YuNetResourceDirector;

// CV_YuNetFramework* CreateYuNetFramework(const char* conf_path){
//     const auto framework=new CV_YuNetFramework(conf_path);
//     auto director= CV_YuNetResourceDirector();
//     framework->ResourceDirector(director);
//     return framework;
// }

// void DestroyYuNetFramework(const CV_YuNetFramework* framework){
//     delete framework;
// }
CV_YuNetFramework* gdlz::cv_framework::face::yunet_export::CreateYuNetFramework(const char* conf_path)
{
    const auto framework=new CV_YuNetFramework(conf_path);
    auto director= CV_YuNetResourceDirector();
    framework->ResourceDirector(director);
    return framework;
}

void gdlz::cv_framework::face::yunet_export::DestroyYuNetFramework(const CV_YuNetFramework* framework)
{
    delete framework;
}

void gdlz::cv_framework::face::yunet_export::YuNetDetection(
    CV_YuNetFramework* framework,
    int size, unsigned char* byte,
    data::YunetOutput& output
 ){
    auto param_=param::YunetParam();
    framework->Input(size,byte,param_).
        Process(param_).
        GetOutput(output,param_);

}

void gdlz::cv_framework::face::yunet_export::CV_YuNetDetection_FeatureVec_Arc(
   CV_YuNetFramework* framework,
   arcface::CV_ArcFaceFramework* arcface_framework,
   int size, unsigned char* byte,
   data::YunetOutput& output0,
   data::FeatureOutput& output1
   )
{

    auto param_=param::YunetParam();
    framework->Input(size,byte,param_).
        Process(param_).
        GetOutput(output0,param_);
     arcface_framework->Process(param_,output1);
}
