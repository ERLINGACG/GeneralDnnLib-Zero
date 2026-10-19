//
// Created by HP on 2026/11/2.
//

#ifndef GENERALDNNLIB_ZERO_TRT_CUDA_BUFFER_TYPE_H
#define GENERALDNNLIB_ZERO_TRT_CUDA_BUFFER_TYPE_H
#include <variant>

namespace gdlz::trt::core::cuda {

    struct kFloat32 {using Type = float;   };

    struct kFloat16 {using Type = uint16_t;};

    struct kDouble  {using Type = double;  };

    struct kInt32   {using Type = int32_t; };

    struct kInt64   {using Type = int64_t; };

    using BufferType = std::variant<kFloat32, kFloat16, kDouble, kInt32, kInt64>;
}
#endif //GENERALDNNLIB_ZERO_TRT_CUDA_BUFFER_TYPE_H