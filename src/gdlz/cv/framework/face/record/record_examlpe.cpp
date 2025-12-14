//
// Created by HP on 2026/1/14.
//

#include "gdlz/export.h"
#include "gdlz/cv/framework/face/arcface/arcface_framework.h"
#include "gdlz/cv/framework/face/utils/record.h"
#include "gdlz/cv/framework/face/yunet/yunet_framework.h"
using namespace gdlz::cv_framework::utils;
using namespace gdlz::cv_framework::face::yunet;
using namespace gdlz::cv_framework::face::arcface;
GDLZ_CORE_API void AssemblyFaceRecord(
    FaceRecord<CV_YuNetFramework,CV_ArcFaceFramework>& record,
    CV_YuNetFramework* yunet_framework,
    CV_ArcFaceFramework* arcface_framework
){
    record.Face_Detector = std::unique_ptr<CV_YuNetFramework>(yunet_framework);
    record.Face_Feature_Extractor = std::unique_ptr<CV_ArcFaceFramework>(arcface_framework);
}
GDLZ_CORE_API void ClearFaceRecord(
    FaceRecord<CV_YuNetFramework,CV_ArcFaceFramework>& record
){
    record.Clear();
}

GDLZ_CORE_API void DeleteFaceRecord(
    FaceRecord<CV_YuNetFramework,CV_ArcFaceFramework>& record
){
    // delete record;
    record.Face_Detector.reset();
    record.Face_Feature_Extractor.reset();
}
GDLZ_CORE_API void AssemblyFaceRecord_1(
    FaceRecord<CV_YuNetFramework,CV_ArcFaceFramework>& record,
    const char* conf_path_1,
    const char* conf_path_2
    // CV_YuNetFramework* yunet_framework,
    // CV_ArcFaceFramework* arcface_framework
){
    auto director_0 = gdlz::resources::CV_YuNetResourceDirector();
    auto director_1 = gdlz::resources::CV_FeatureDnnResourceDirector();

    record.Face_Detector = std::make_unique<CV_YuNetFramework>(conf_path_1);
    record.Face_Detector->ResourceDirector(director_0);

    record.Face_Feature_Extractor = std::make_unique<CV_ArcFaceFramework>();
    record.Face_Feature_Extractor->SetConfPath(conf_path_2);
    record.Face_Feature_Extractor->ResourceDirector(director_1);

}


GDLZ_CORE_API void ExampleFaceRecordDo(
    const FaceRecord<CV_YuNetFramework,CV_ArcFaceFramework>& record,
    int size, unsigned char* data,
    gdlz::cv_framework::data::YunetOutput&   output1,
    gdlz::cv_framework::data::FeatureOutput& output2
){
    try
    {
        auto param = gdlz::cv_framework::param::YunetParam();
        record.Face_Detector->Input(size,data,param).Process(param).GetOutput(output1,param);
        record.Face_Feature_Extractor->Process(param,output2);

    }catch (std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }


}
