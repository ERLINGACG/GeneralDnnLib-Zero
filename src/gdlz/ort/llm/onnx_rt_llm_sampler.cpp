//
// Created by HP on 2026/5/16.
//
#include "gdlz/ort/llm/onnx_rt_llm_sampler.h"

#include <numeric>


void gdlz::ort::llm::sampler::OnnxRtLLmSampler::SetParam(const OnnxRtLLmParam& param)
{
    this->rt_param=param;
    this->rng.seed(this->rt_param.seed);
}

std::vector<float> gdlz::ort::llm::sampler::OnnxRtLLmSampler::softmax_with_temperature(const std::vector<float>& logits,
    float temperature)
{
    std::vector<float> probs(logits.size()); //准备数组
    float max_logit = *std::ranges::max_element(logits); //找最大值
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        float val = (logits[i] - max_logit) / temperature;             // 3. 减去最大值（数值稳定）再除以温度
        probs[i] = std::exp(val);                                      // 4. 计算 e^(val)
        sum += probs[i];                                               // 5. 累加求和
    }
    for (float& p : probs) p /= sum;                                    // 6. 每个值除以总和，得到概率

    return probs;
}

std::vector<float> gdlz::ort::llm::sampler::OnnxRtLLmSampler::apply_top_k_filter(const std::vector<float>& probs, int32_t top_k)
{
    if (top_k <= 0) return probs;
    std::vector<size_t> idx(probs.size());        // 准备索引数组,映射到概率数组的索引
    std::iota(idx.begin(), idx.end(), 0); // 填充索引数组
    std::partial_sort(                                // 偏序排序
        idx.begin(), idx.begin() + top_k, idx.end(),
        [&](size_t a, size_t b) {
           return probs[a] > probs[b];
       }
    );
    std::vector filtered(probs.size(), 0.0f);
    float sum = 0.0f;
    for (int i = 0; i < top_k; ++i) {
        filtered[idx[i]] = probs[idx[i]];
        sum += filtered[idx[i]];
    }
    // 3. 重新归一化
    if (sum > 0.0f) {
        for (float& p : filtered) p /= sum;
    }


    return filtered;
}

std::vector<float> gdlz::ort::llm::sampler::OnnxRtLLmSampler::apply_top_p_filter(const std::vector<float>& probs, float top_p)
{
    // 如果 top_p 大于等于1，则直接返回原始概率
    if (top_p >= 1.0f - 1e-6f) {
        return probs;
    }

    // 如果 top_p 极小（<=0），则只保留概率最大的一个词（退化到贪心）
    if (top_p <= 0.0f) {
        std::vector<float> filtered(probs.size(), 0.0f);
        auto max_it = std::ranges::max_element(probs);
        if (max_it != probs.end()) {
            filtered[std::distance(probs.begin(), max_it)] = 1.0f;
        }
        return filtered;
    }

    std::vector<size_t> indices(probs.size());
    std::iota(indices.begin(), indices.end(), 0); // 填充索引数组
    std::ranges::sort(indices,[&](size_t i, size_t j) { return probs[i] > probs[j]; }); // 降序排序

    float cum = 0.0f;
    size_t keep_cnt = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        cum += probs[indices[i]];
        keep_cnt=i+1; // 至少保留一个
        if (cum >= top_p) {
            break;
        }
    }

    // 3. 构造过滤后的概率向量（未归一化）
    std::vector filtered(probs.size(), 0.0f);
    float sum = 0.0f;
    for (size_t i = 0; i < keep_cnt; ++i) {
        const size_t idx = indices[i];
        filtered[idx] = probs[idx];
        sum += filtered[idx];
    }

    // 4. 重归一化
    if (sum > 0.0f) {
        const float inv_sum = 1.0f / sum;
        for (float& p : filtered) {
            p *= inv_sum;
        }
    }

    return filtered;
}

int64_t gdlz::ort::llm::sampler::OnnxRtLLmSampler::sample_from_probs(const std::vector<float>& probs,std::mt19937& rng)
{
    if (probs.empty()) {
        return -1;  // 错误情况
    }
    // 使用 discrete_distribution，它接受权重范围（不必归一化，但已经归一化也无妨）
    std::discrete_distribution<int64_t> dist(probs.begin(), probs.end());
    return dist(rng);
}

int64_t gdlz::ort::llm::sampler::OnnxRtLLmSampler::SampleToken(const std::vector<Ort::Value>& value) const
{
    int64_t token=0;

    const auto& logits = value[0];
    const auto shape = logits.GetTensorTypeAndShapeInfo().GetShape();
    auto* logits_data = logits.GetTensorData<float>();
    constexpr int64_t batch_idx = 0;
    const int64_t last_pos = shape[1] - 1;
    const int64_t vocab_size = shape[2];
    const float* last_token_logits = logits_data + (batch_idx * shape[1] + last_pos) * vocab_size;

    std::vector logits_vec(last_token_logits, last_token_logits + vocab_size);

    auto max_element=[&]() {
        token=std::distance(
            last_token_logits,
            std::max_element(last_token_logits, last_token_logits + vocab_size)
        );
    };

    auto top_k_sample=[&](std::vector<float>& probs) {
        probs = apply_top_k_filter(probs, rt_param.top_k);
    };

    auto top_p_sample=[&](std::vector<float>& probs) {
        probs = apply_top_p_filter(probs, rt_param.top_p);
    };


    auto temperature_sample=[&](std::vector<float>& probs) {
        probs = softmax_with_temperature(probs, rt_param.temperature);
    };


    if (rt_param.temperature==0) {
        max_element();
    }else {
        temperature_sample(logits_vec);
        top_k_sample(logits_vec);
        top_p_sample(logits_vec);
        token=sample_from_probs(logits_vec, rng);
    }

    return token;
}
