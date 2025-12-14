//
// Created by HP on 2026/1/6.
//

#ifndef GENERALDNNLIB_ZERO_YU_NET_FRAMEWORK_H
#define GENERALDNNLIB_ZERO_YU_NET_FRAMEWORK_H
#include "yunet_param.h"
#include "yunet_resources.h"
#include "gdlz/export.h"
#include "gdlz/cv/data/yunet_out_put.h"
#include "gdlz/cv/framework/face/base/feature_dnn.h"


namespace gdlz::cv_framework::face::yunet
{

     class GDLZ_CPP_API CV_YuNetFramework
     {
          resources::YuNetResource resource{};
          std::string   config_path;

          public:
               explicit CV_YuNetFramework(std::string  config_path):config_path(std::move(config_path)){};
               ~CV_YuNetFramework() =default;

               CV_YuNetFramework& ResourceDirector(resources::CV_YuNetResourceDirector& director);
               CV_YuNetFramework& Input(int size, unsigned char* byte,param::YunetParam& param);
               CV_YuNetFramework& Process(param::YunetParam& param);
               CV_YuNetFramework& GetOutput(data::YunetOutput& output, const param::YunetParam& param);

     };
}
#endif //GENERALDNNLIB_ZERO_YU_NET_FRAMEWORK_H
