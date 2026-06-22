//
// Created by HP on 2026/6/13.
//
#include "gdlz/trt/illm/sample/sp/trt_illm_sample.h"

#include "gdlz/trt/illm/sample/cu/trt_illm_temp.cuh"

int64_t gdlz::trt::illm::sample::TensorRTIllmSample::SampleTemperature(void const* logits, int64_t start_token_pos,
                                                                       int64_t loca_v
) {
    return cu::Argmax(logits,start_token_pos,loca_v);
}

int64_t gdlz::trt::illm::sample::TensorRTIllmSample::SampleTemperature(void const* logits, int64_t start_token_pos,
                                                                       int64_t loca_v, int64_t*& output_address)
{
   return  cu::Argmax(logits,start_token_pos,loca_v,output_address);
}
