//
// Created by HP on 2025/12/9.
//

#ifndef YOLO_RESOURCES_H
#define YOLO_RESOURCES_H
#include <nlohmann/json.hpp>

#include "gdlz/export.h"

#include "gdlz/res/resources.h"

namespace gdlz::resources
{
    // YOLO 资源类型
    struct  YoloResource : ResourceBase
    {
        std::string model_path;
        int input_height;
        int input_width;
        int enable_backend;
        float conf_thresh;
        float nms_thresh;
        bool is_fusion;
        std::unique_ptr<std::vector<std::string>> class_names= std::make_unique<std::vector<std::string>>();
    };

    class GDLZ_CPP_API CV_YoloResourceDirector : public ResourceDirectorBase<YoloResource,CV_YoloResourceDirector>
    {

        public:
            std::string conf_path;

            ~CV_YoloResourceDirector() override =default;
            CV_YoloResourceDirector& Hand(YoloResource& resource) override;
            CV_YoloResourceDirector& SetConfPath(const std::string& conf_path);
    };


} // namespace gdlz::resources
#endif //YOLO_RESOURCES_H
