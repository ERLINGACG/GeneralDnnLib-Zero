//
// Created by HP on 2026/1/12.
//

#ifndef GENERALDNNLIB_ZERO_YUNET_OUT_PUT_H
#define GENERALDNNLIB_ZERO_YUNET_OUT_PUT_H
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>

#include "feature_output.h"

namespace gdlz::cv_framework::data
{
    struct YunetOutput
    {
        std::unique_ptr<unsigned char[]> byte_data{};
        int size{};

        auto& SetData(const cv::Mat& mat)
        {
            const std::vector params_ {cv::IMWRITE_WEBP_QUALITY, 95};
            std::vector<unsigned char> encoded;
            cv::imencode(".jpeg", mat, encoded, params_);
            size =  static_cast<int>(encoded.size());
            byte_data = std::make_unique<unsigned char[]>(encoded.size());
            memcpy(byte_data.get(), encoded.data(), encoded.size());


            // auto buffer = std::make_unique<char[]>(json_str_.size() + 1);
            // std::ranges::copy(json_str_, buffer.get());
            // buffer[json_str_.size()] = '\0'; // 手动添加 null 终止符
            // json_str=std::move(buffer);
            // json_size = static_cast<int>(json_str_.size());
            return *this;
        }
    };
}
#endif //GENERALDNNLIB_ZERO_YUNET_OUT_PUT_H