//
// Created by HP on 2026/1/14.
//
#include "gdlz/cv/framework/face/arcface/arcface_framework.h"
#include <opencv2/opencv.hpp>
using gdlz::cv_framework::param::FaceParam;
using namespace gdlz::cv_framework::face::arcface;
CV_ArcFaceFramework& CV_ArcFaceFramework::Process(FaceParam& param, data::FeatureOutput& feature_output)
{
    try{
        cv::Mat alignedFace;
        auto facesRect = std::move(param.rects);
        auto vec_feature_output=std::vector<cv::Mat>();
        for (int i = 0; i < facesRect->size(); i++) {
            auto face = (*facesRect)[i];
            auto faceROI = (*param.input)(face);
            if (faceROI.empty()) {
                continue;
            }
            std::vector<cv::Point2f> currentLandmarks;
            for (int j = 0; j < 5; ++j) {
                const float x = param.faces->at<float>(i, 4 + 2 * j);  // 第i张脸的第j个关键点x
                const float y = param.faces->at<float>(i, 5 + 2 * j);  // 第i张脸的第j个关键点y
                // 转换为ROI内的相对坐标（减去人脸框左上角）
                currentLandmarks.emplace_back(x - static_cast<float>(face.x), y - static_cast<float>(face.y));

            }
            cv::Mat affineMatrix = cv::estimateAffinePartial2D(
                currentLandmarks,  // 输入：当前人脸关键点
                standardLandmarks,  // 输出：标准关键点
                cv::noArray()
            );

            cv::warpAffine(
                faceROI,                // 输入：原始人脸ROI
                alignedFace,            // 输出：对齐后的人脸
                affineMatrix,           // 仿射矩阵
                cv::Size(this->resource.input_width, this->resource.input_height),     // 对齐后的尺寸（需与标准关键点匹配）
                cv::INTER_LINEAR,       // 插值方式
                cv::BORDER_CONSTANT,    // 边界填充方式
                cv::Scalar(0, 0, 0)     // 边界填充颜色
            );
            if (alignedFace.empty()) {
                continue;
            }
            this->resource.Net->setInput(
                cv::dnn::blobFromImage(
                alignedFace,
                1.0,
                cv::Size(
                    this->resource.input_width,
                    this->resource.input_height
                ),
                cv::Scalar(104, 117, 123),
                false,
                false
            ));
            auto output = this->resource.Net->forward();
            cv::Mat featureVec = output.reshape(1, 1);
            // std::cout << featureVec << std::endl;
            vec_feature_output.push_back(featureVec);
        }
        feature_output.InsertData512(vec_feature_output);
    }catch (const std::exception& e){
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return *this;
}
