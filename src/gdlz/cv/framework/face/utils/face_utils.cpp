//
// Created by HP on 2026/1/17.
//
#include "gdlz/cv/framework/face/utils/face_utils.h"

void gdlz::cv_framework::face::utils::GetFeatureVec512(const unsigned char* feature, const int size)
{
     // int numBytes = size % 512 * sizeof(float);
     // auto featureVec = std::vector<float>();
     // 创建1024个float的vector
     std::vector<float> featureVec(size / 4);

     // 将字节数据复制到float数组中
     // 注意：这里假设字节顺序与当前平台一致

     std::memcpy(featureVec.data(), feature, size);
     std::cout<<"featureVec.size():"<<featureVec.size()<<std::endl;
     for (int i = 0; i < featureVec.size(); i++)
     {
          std::cout <<"featureVec["<<i<<"]:"<< featureVec[i]<<std::endl;
     }

}

double gdlz::cv_framework::face::utils::CompareVec512(unsigned char* feature1, unsigned char* feature2)
{
     auto* feat1 = reinterpret_cast<float*>(feature1);
     auto* feat2 = reinterpret_cast<float*>(feature2);

     cv::Mat vec1(1, 512, CV_32F, feat1);
     cv::Mat vec2(1, 512, CV_32F, feat2);

     normalize(vec1, vec1, 1.0, 0.0, cv::NORM_L2);  // 归一化到模长为1
     normalize(vec2, vec2, 1.0, 0.0, cv::NORM_L2);
     return norm(vec1 - vec2);
}
