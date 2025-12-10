//
// Created by HP on 2025/12/10.
//

#ifndef OUT_PUT_H
#define OUT_PUT_H
#include <opencv2/imgcodecs.hpp>

namespace gdlz::cv_framework::data
{
    struct YoloOutput
    {
        std::unique_ptr<unsigned char[]> byte{};
        std::unique_ptr<char[]> json_str{};
        int json_size{};
        int size{};


        auto& SetData(const cv::Mat& mat,const std::string& json_str_)
        {
            const std::vector params_ {cv::IMWRITE_WEBP_QUALITY, 95};
            std::vector<unsigned char> encoded;
            cv::imencode(".jpeg", mat, encoded, params_);
            size =  static_cast<int>(encoded.size());
            byte = std::make_unique<unsigned char[]>(encoded.size());
            memcpy(byte.get(), encoded.data(), encoded.size());


            auto buffer = std::make_unique<char[]>(json_str_.size() + 1);
            std::ranges::copy(json_str_, buffer.get());
            buffer[json_str_.size()] = '\0'; // 手动添加 null 终止符
            json_str=std::move(buffer);
            json_size = static_cast<int>(json_str_.size());
            return *this;
        }
    };
}
#endif //OUT_PUT_H
