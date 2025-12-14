//
// Created by HP on 2026/1/14.
//

#ifndef GENERALDNNLIB_ZERO_FEATURE_OUTPUT_H
#define GENERALDNNLIB_ZERO_FEATURE_OUTPUT_H
namespace gdlz::cv_framework::data
{
    struct FeatureOutput
    {
        std::unique_ptr<unsigned char[]> feature_data;
        int size;


        void SetData(const cv::Mat& mat)
        {
            if (mat.empty() || mat.dims != 2 || mat.rows != 1)
            {
                feature_data.reset(); // 清空原有数据
                size = 0;
                return ;
            }

            size = mat.total() * mat.elemSize();
            // 3. 申请对应大小的内存：std::make_unique自动初始化，内存安全
            feature_data = std::make_unique<unsigned char[]>(size);
            memcpy(feature_data.get(), mat.data, size);

        }

        void InsertData512(const std::vector<cv::Mat>& data)
        {

            size = static_cast<int>(512 * sizeof(float) * data.size());
            auto new_data = std::make_unique<unsigned char[]>(size);
            size_t offset = 0;
            for (const auto & i : data)
            {
                const size_t copySize = i.total() * i.elemSize();
                memcpy(
                    new_data.get() + offset,  //起始数据+偏移量
                    i.data,                  //源数据
                    copySize                 //复制大小
                    );
                offset += copySize;         //偏移量增加复制大小
            }
            feature_data = std::move(new_data);
        }

        void InsertData128(const std::vector<cv::Mat>& data)
        {
            size_t offset = 0;
            size = static_cast<int>(128 * sizeof(float) * data.size());
            auto new_data = std::make_unique<unsigned char[]>(size);
            for (const auto & i : data)
            {
                const size_t copySize = i.total() * i.elemSize();
                memcpy(
                    new_data.get() + offset,  //起始数据+偏移量
                    i.data,                  //源数据
                    copySize                 //复制大小
                    );
                offset += copySize;         //偏移量增加复制大小
            }
            feature_data = std::move(new_data);
        }
    };
} // namespace gdlz::cv_framework::face::base
#endif //GENERALDNNLIB_ZERO_FEATURE_OUTPUT_H