//
// Created by HP on 2026/1/13.
//

#ifndef GENERALDNNLIB_ZERO_FEATURE_DNN_H
#define GENERALDNNLIB_ZERO_FEATURE_DNN_H
#include "feature_dnn_resources_director.h"

namespace gdlz::cv_framework::face::base
{
     template<typename T>
     class CV_FeatureDnnBase
     {

          protected:
            std::string conf_path;
            resources::FeatureDnnResource resource{};
          public:
                ~CV_FeatureDnnBase() = default;
                 T& ResourceDirector(resources::CV_FeatureDnnResourceDirector& director)
                 {
                     director.SetConfPath(conf_path).Hand(resource);
                     return static_cast<T&>(*this);
                 };

                 T& SetConfPath(const std::string& conf_path)
                 {
                     this->conf_path=conf_path;
                     return static_cast<T&>(*this);
                 };
     };
}
#endif //GENERALDNNLIB_ZERO_FEATURE_DNN_H