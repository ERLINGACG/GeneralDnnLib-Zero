//
// Created by HP on 2026/1/13.
//

#ifndef GENERALDNNLIB_ZERO_FEATURE_DNN_RESOURCES_H
#define GENERALDNNLIB_ZERO_FEATURE_DNN_RESOURCES_H
#include "gdlz/res/resources.h"

namespace gdlz::resources
{
        struct FeatureDnnResource : ResourceBase
        {
            std::string model_path;
            int input_height;
            int input_width;
            int enable_backend;
            bool is_fusion;

        };
        class CV_FeatureDnnResourceDirector : public ResourceDirectorBase<FeatureDnnResource,CV_FeatureDnnResourceDirector>
        {
            std::string conf_path;
            public:
                ~CV_FeatureDnnResourceDirector() override = default;
                CV_FeatureDnnResourceDirector& Hand(FeatureDnnResource& resource) override;
                CV_FeatureDnnResourceDirector& SetConfPath(const std::string& conf_path);
        };
}
#endif //GENERALDNNLIB_ZERO_FEATURE_DNN_RESOURCES_H