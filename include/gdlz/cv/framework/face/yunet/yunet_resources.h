//
// Created by HP on 2026/1/6.
//

#ifndef GENERALDNNLIB_ZERO_YU_NET_RESOURCES_H
#define GENERALDNNLIB_ZERO_YU_NET_RESOURCES_H
#include "gdlz/res/resources.h"
#include "gdlz/export.h"
#include <opencv2/objdetect/face.hpp>
namespace gdlz::resources
{
        struct YuNetResource
        {
            cv::Ptr<cv::FaceDetectorYN> detector;
            std::string model_path;
            int input_height;
            int input_width;
            int enable_backend;
            float conf_thresh;
            float nms_thresh;
            int max_Face;
            bool is_fusion;
            int  backend;
        };

        class  GDLZ_CPP_API CV_YuNetResourceDirector : resources::ResourceDirectorBase<YuNetResource,CV_YuNetResourceDirector> {
            std::string conf_path;
            public:
                    ~CV_YuNetResourceDirector() override =default;
                    CV_YuNetResourceDirector& Hand(YuNetResource& resource) override;
                    CV_YuNetResourceDirector& SetConfigPath(const std::string& config_path);
        };
}

#endif //GENERALDNNLIB_ZERO_YU_NET_RESOURCES_H
