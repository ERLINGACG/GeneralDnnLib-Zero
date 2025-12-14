//
// Created by HP on 2026/1/6.
//

#include "gdlz/cv/framework/face/yunet/yunet_framework.h"

#include <dylog/logger.h>
using gdlz::cv_framework::face::yunet::CV_YuNetFramework;

CV_YuNetFramework& CV_YuNetFramework::ResourceDirector(resources::CV_YuNetResourceDirector& director)
{
    director.SetConfigPath(this->config_path).Hand(this->resource);
    return *this;
}

CV_YuNetFramework& CV_YuNetFramework::Input(int size, unsigned char* byte, param::YunetParam& param)
{
    const std::vector buf(byte,byte+size);
    param.input=std::make_unique<cv::Mat>(cv::imdecode(buf, cv::IMREAD_COLOR));


    this->resource.detector->setInputSize(param.input->size());
    return *this;
}

CV_YuNetFramework& CV_YuNetFramework::Process(param::YunetParam& param)
{
    try{
        param.faces=std::make_unique<cv::Mat>(*new cv::Mat());
        this->resource.detector->detect(*param.input,*param.faces);
        for (int i = 0; i < param.faces->rows; i++)
        {
            dylog::DynamicLogger().debug("face: {}",i);
            const int x = static_cast<int>(param.faces->at<float>(i, 0) );
            const int y = static_cast<int>(param.faces->at<float>(i, 1) );
            const int w = static_cast<int>(param.faces->at<float>(i, 2) );
            const int h = static_cast<int>(param.faces->at<float>(i, 3) );
            cv::Rect faceRect(x, y, w, h);
            cv::rectangle(*param.input, faceRect, cv::Scalar(0, 255, 0), 2);
            param.rects->push_back(faceRect);
        }
    }catch (const std::exception& e){
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return *this;
}

CV_YuNetFramework& CV_YuNetFramework::GetOutput(data::YunetOutput& output, const param::YunetParam& param)
{
    output.SetData(*param.input);
    return *this;
}


