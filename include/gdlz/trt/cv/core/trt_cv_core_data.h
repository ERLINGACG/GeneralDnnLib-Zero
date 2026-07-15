//
// Created by HP on 2026/7/8.
//

#ifndef GENERALDNNLIB_ZERO_TRT_CV_CORE_DATA_H
#define GENERALDNNLIB_ZERO_TRT_CV_CORE_DATA_H
#include "gdlz/trt/core/trt_core_data.h"

namespace gdlz::trt::icv::core{


    struct TensorRTCvEngine{

        trt::core::TensorRTCoreEngine engine{};

        auto& getCoreEngine(){
            return engine;
        }

    };

    struct TensorRTCvCtx{
        std::unique_ptr<trt::core::TensorRTCoreCtx>        core_ctx;
        std::unique_ptr<std::map<std::string,std::string>> layer_names;
        std::unique_ptr<std::map<std::string,void*>> layer_address;
        [[nodiscard]] auto& getCoreCtx() const{
            return *core_ctx;
        }
    };

}
#endif //GENERALDNNLIB_ZERO_TRT_CV_CORE_DATA_H