//
// Created by HP on 2026/6/12.
//

#ifndef GENERALDNNLIB_ZERO_TRT_ILLM_CU_PARAM_CUH
#define GENERALDNNLIB_ZERO_TRT_ILLM_CU_PARAM_CUH
#include <cstdint>

namespace gdlz::trt::illm::sample::cu{

    struct LLmCuParam{
        int64_t batch;
        int64_t token_pos;
        int64_t vocab_size;
    };

    struct LLmKVCuParam{
        int64_t batch;
        int64_t heads;
        int64_t head_dims;
        int64_t start_token_pos;
        int64_t end_token_pos;
    };
}
#endif //GENERALDNNLIB_ZERO_TRT_ILLM_CU_PARAM_CUH