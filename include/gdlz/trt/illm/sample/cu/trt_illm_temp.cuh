//
// Created by HP on 2026/6/12.
//

#ifndef GENERALDNNLIB_ZERO_TRT_ILLM_TEMP_CUH
#define GENERALDNNLIB_ZERO_TRT_ILLM_TEMP_CUH
#include <cstdint>

namespace gdlz::trt::illm::sample::cu{

    int Argmax(const void* logits,int64_t start_token_pos,int64_t loca_v);
    int64_t Argmax(const void* logits, int64_t start_token_pos, int64_t loca_v, int64_t*& output_address);

}
#endif //GENERALDNNLIB_ZERO_TRT_ILLM_TEMP_CUH