//
// Created by HP on 2025/12/11.
//

#ifndef SR_FRAMEWORK_H
#define SR_FRAMEWORK_H
#include <string>

#include "sr_resources_director.h"
#include "gdlz/export.h"
#include "gdlz/cv/framework/param.h"

namespace gdlz::cv_framework::sr
{
    class GDLZ_CPP_API SrFramework {
        resources::SrResource resource{};
        std::string conf_path;
    public:
        explicit SrFramework(const std::string& conf_path):conf_path(std::move(conf_path)){};
        ~SrFramework() = default;

        SrFramework& ResourceDirector(resources::SrResourceDirector& resource_director);

        SrFramework& Upsample(const param::BaseParam& param);
    };
}
#endif //SR_FRAMEWORK_H
