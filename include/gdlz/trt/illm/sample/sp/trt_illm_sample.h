//
// Created by HP on 2026/6/12.
//

#ifndef GENERALDNNLIB_ZERO_TRT_ILLM_SAMPLE_H
#define GENERALDNNLIB_ZERO_TRT_ILLM_SAMPLE_H
#include <cstdint>

#include "gdlz/export.h"

namespace gdlz::trt::illm::sample {

    struct TensorRTIllmSampleParams{
        int64_t top_k;
        float   top_p;
        float   temp;
    };

    class GDLZ_CPP_API TensorRTIllmSample {
          TensorRTIllmSampleParams params{};
        public:
            TensorRTIllmSample (TensorRTIllmSampleParams params) : params(params) {}
            ~TensorRTIllmSample() = default;

            int64_t SampleTemperature(void const* logits, int64_t start_token_pos, int64_t loca_v);
            int64_t SampleTemperature(void const* logits, int64_t start_token_pos, int64_t loca_v, int64_t*& output_address);
    };
}
#endif //GENERALDNNLIB_ZERO_TRT_ILLM_SAMPLE_H