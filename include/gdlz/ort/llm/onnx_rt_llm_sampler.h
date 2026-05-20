//
// Created by HP on 2026/5/18.
//

#ifndef GENERALDNNLIB_ZERO_ONNX_RT_LLM_SAMPLER_H_H
#define GENERALDNNLIB_ZERO_ONNX_RT_LLM_SAMPLER_H_H
#include <random>
#include <vector>

// #include "gdlz/ort/llm/onnx_rt_llm_data.h"
#include <onnxruntime_cxx_api.h>

#include "gdlz/export.h"
namespace gdlz::ort::llm::sampler {

    struct OnnxRtLLmParam{
        int32_t top_k;        // top_k采样
        float_t top_p;       // top_p采样
        float_t temperature;// 温度
        int64_t seed;      // 随机种子
    };

    class GDLZ_CPP_API OnnxRtLLmSampler {

        OnnxRtLLmParam rt_param{};
        mutable std::mt19937 rng;

    public:
        OnnxRtLLmSampler()=default;
        ~OnnxRtLLmSampler() = default;

        void SetParam(const OnnxRtLLmParam& param);

        [[nodiscard]] static std::vector<float> softmax_with_temperature(const std::vector<float>& logits, float temperature);
        [[nodiscard]] static std::vector<float> apply_top_k_filter(const std::vector<float>& probs, int32_t top_k);
        [[nodiscard]] static std::vector<float> apply_top_p_filter(const std::vector<float>& probs, float top_p);
        [[nodiscard]] static int64_t            sample_from_probs(const std::vector<float>& probs, std::mt19937& rng);
        [[nodiscard]] int64_t SampleToken(const std::vector<Ort::Value>& value) const;
    };
}
#endif //GENERALDNNLIB_ZERO_ONNX_RT_LLM_SAMPLER_H_H