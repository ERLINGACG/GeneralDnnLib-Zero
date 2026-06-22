//
// Created by HP on 2026/6/16.
//

#ifndef GENERALDNNLIB_ZERO_TRT_ILLM_CONFIG_H
#define GENERALDNNLIB_ZERO_TRT_ILLM_CONFIG_H
#include <cstdint>
// #include <opencv2/core/hal/interface.h>

namespace gdlz::trt::illm::config{
    struct TensorRTIllmConfig {
        int32_t enable_position;
    };
}
#endif //GENERALDNNLIB_ZERO_TRT_ILLM_CONFIG_H