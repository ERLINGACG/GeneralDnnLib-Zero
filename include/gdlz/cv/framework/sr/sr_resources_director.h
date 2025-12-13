//
// Created by HP on 2025/12/11.
//

#ifndef SR_RESOURCES_H
#define SR_RESOURCES_H
#include "opencv2/dnn_superres.hpp"
#include "gdlz/export.h"
namespace gdlz::resources
{

    struct SrResource
    {
        std::string model_path;
        std::string type;
        int scale;
        cv::dnn_superres::DnnSuperResImpl sr;
    };

    class GDLZ_CPP_API SrResourceDirector
    {
        public:
            std::string conf_path;

            ~SrResourceDirector() =default;
            SrResourceDirector& Hand(SrResource& resource);
            SrResourceDirector& SetConfPath(const std::string& conf_path);
    };
}
#endif //SR_RESOURCES_H
