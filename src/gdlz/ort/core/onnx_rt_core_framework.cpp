//
// Created by HP on 2026/5/3.
//
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <ostream>
#include <dylog/logger.h>
#include <opencv2/core/hal/interface.h>

#include "gdlz/ort/core/onnx_rt_core_director.h"
#include "gdlz/ort/core/onnx_rt_framework.h"

gdlz::ort::core::OnnxRtFramework* gdlz::ort::core::OnnxRtFramework::ResourceDirector(const char* config_path)
{
    OnnxRtCoreDirector::Handle(engine_info, config_path);
    return this;
}

int gdlz::ort::core::OnnxRtFramework::CreateEngine(data::OnnxRTEngine& engine) const
{
    try
    {
        size_t len = mbstowcs(nullptr, engine_info.model_path.c_str(), 0);  // 鑾峰彇鎵€闇€瀹藉瓧绗﹂暱搴?
        if (len == static_cast<size_t>(-1)) { /* 杞崲澶辫触 */ }
        auto* w_str = static_cast<wchar_t*>(malloc((len + 1) * sizeof(wchar_t)));
        mbstowcs(w_str, engine_info.model_path.c_str(), len + 1);

        engine.sessionOptions = std::make_unique<Ort::SessionOptions>();
        engine.sessionOptions->SetIntraOpNumThreads(1);
        engine.sessionOptions->SetGraphOptimizationLevel(ORT_ENABLE_ALL);


        if (engine_info.use_cuda)
        {
            engine.cudaProviderOptions = std::make_unique<OrtCUDAProviderOptions>();
            engine.cudaProviderOptions->device_id=0;
            engine.sessionOptions->AppendExecutionProvider_CUDA(*engine.cudaProviderOptions);  // 新增这一行

        }
        if (engine.session==nullptr){
            engine.session =std::make_unique<Ort::Session>(
                engine_info.env, w_str, *engine.sessionOptions
            );
            free(w_str);
            if (engine.session!=nullptr){
                dylog::DynamicLogger().setInvokeName("OnnxRtFramework::CreateEngine").info("OnnxRtFramework::CreateEngine, session created");
            }
        }

        if (engine.input_name_ptr==nullptr || engine.output_name_ptr==nullptr &&  engine.session!=nullptr) {
            // engine.input_name_ptr = engine.session->GetInputNames();
            engine.input_name_ptr  = std::make_unique<std::vector<std::string>>();
            engine.output_name_ptr = std::make_unique<std::vector<std::string>>();

            Ort::AllocatorWithDefaultOptions allocator;
            const size_t input_count = engine.session->GetInputCount();
            const size_t output_count = engine.session->GetOutputCount();

            for (size_t i = 0; i < input_count; ++i) {
                engine.input_name_ptr->push_back(engine.session->GetInputNameAllocated(i, allocator).get());
            }
            for (size_t i = 0; i < output_count; ++i) {
                engine.output_name_ptr->push_back(engine.session->GetOutputNameAllocated(i, allocator).get());
            }
        }

        // engine.head_dim = engine_info.head_dim;
        // engine.heads    = engine_info.heads;
        // engine.layers   = engine_info.layers;
        dylog::DynamicLogger().setInvokeName("OnnxRtFramework::CreateCtxV1").info("OnnxRtFramework::CreateEngine success");

    }catch(std::exception& e) {
        dylog::DynamicLogger().error("OnnxRtFramework::CreateEngine, error: {}", e.what());
        return -1;
    }
    return 0;
}

void gdlz::ort::core::OnnxRtFramework::GetEngineInfo(data::OnnxRTEngine& ctx)
{
    if (!ctx.session) return;

    Ort::AllocatorWithDefaultOptions allocator;

    const auto format_shape = [](const std::vector<int64_t>& shape) {
        std::string shape_str = "[";
        for (size_t index = 0; index < shape.size(); ++index) {
            shape_str += std::to_string(shape[index]);
            if (index + 1 != shape.size()) {
                shape_str += ", ";
            }
        }
        shape_str += "]";
        return shape_str;
    };
    const auto format_data_type = [](ONNXTensorElementDataType data_type) {
        switch (data_type) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED: return std::string("undefined");
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return std::string("float");
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return std::string("uint8");
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return std::string("int8");
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return std::string("uint16");
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return std::string("int16");
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return std::string("int32");
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return std::string("int64");
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: return std::string("string");
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return std::string("bool");
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return std::string("float16");
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return std::string("double");
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: return std::string("uint32");
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: return std::string("uint64");
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64: return std::string("complex64");
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128: return std::string("complex128");
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return std::string("bfloat16");
            default: return std::string("unknown");
        }
    };

    const size_t numInputs = ctx.session->GetInputCount();
    const size_t numOutputs = ctx.session->GetOutputCount();

    for (size_t i = 0; i != numInputs; ++i) {
        auto input_name_ptr = ctx.session->GetInputNameAllocated(i, allocator);
        std::string input_name(input_name_ptr.get());

        auto type_info = ctx.session->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        const std::vector<int64_t> input_shape = tensor_info.GetShape();
        const auto input_data_type = tensor_info.GetElementType();
        dylog::DynamicLogger().info(
            "OnnxRtFramework::GetEngineInfo, input name: {}, shape: {}, data_type: {}",
            input_name.c_str(),
            format_shape(input_shape).c_str(),
            format_data_type(input_data_type).c_str()
        );
    }

    for (size_t i = 0; i != numOutputs; ++i) {
        auto output_name_ptr = ctx.session->GetOutputNameAllocated(i, allocator);
        std::string output_name(output_name_ptr.get());

        auto type_info = ctx.session->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        const std::vector<int64_t> output_shape = tensor_info.GetShape();
        const auto output_data_type = tensor_info.GetElementType();
        dylog::DynamicLogger().info(
            "OnnxRtFramework::GetEngineInfo, output name: {}, shape: {}, data_type: {}",
            output_name.c_str(),
            format_shape(output_shape).c_str(),
            format_data_type(output_data_type).c_str()
        );
    }
}

void gdlz::ort::core::OnnxRtFramework::run(data::OnnxRTEngine& ctx) const
{
    if (!ctx.session) return;

    Ort::AllocatorWithDefaultOptions allocator;
    const size_t numInputs = ctx.session->GetInputCount();
    const size_t numOutputs = ctx.session->GetOutputCount();

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

    for (size_t i = 0; i != numInputs; ++i) {
        auto input_name_ptr = ctx.session->GetInputNameAllocated(i, allocator);
        input_names.emplace_back(input_name_ptr.get());
    }

    for (size_t i = 0; i != numOutputs; ++i) {
        auto output_name_ptr = ctx.session->GetOutputNameAllocated(i, allocator);
        output_names.emplace_back(output_name_ptr.get());
    }

    // 初始 prompt tokens（示例）
    std::vector<int64_t> input_ids = {
        151646, 151644, 108386, 103924, 11, 14880, 100157, 107828, 198, 151645, 198
    };
    int64_t current_seq_len = static_cast<int64_t>(input_ids.size());
    constexpr int64_t batch_size = 1;
    constexpr int max_new_tokens = 100;

    // 第一次推理所需的 attention_mask 和 position_ids（完整序列）
    std::vector<int64_t> attention_mask(current_seq_len, 1);
    std::vector<int64_t> position_ids(current_seq_len);
    for (int64_t i = 0; i < current_seq_len; ++i) position_ids[i] = i;

    std::vector<int64_t> generated_tokens;
    generated_tokens.reserve(max_new_tokens);

    // 使用 CPU 内存（可根据需要改为 GPU）
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // 创建第一次推理的输入张量
    std::vector input_shape = {batch_size, current_seq_len};
    auto input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
        mem_info, input_ids.data(), input_ids.size(),
        input_shape.data(), input_shape.size());
    auto attn_mask_tensor = Ort::Value::CreateTensor<int64_t>(
        mem_info, attention_mask.data(), attention_mask.size(),
        input_shape.data(), input_shape.size());
    auto pos_ids_tensor = Ort::Value::CreateTensor<int64_t>(
        mem_info, position_ids.data(), position_ids.size(),
        input_shape.data(), input_shape.size());

    // 创建空的 past_key_values（28层，每层 key+value）
    const int NUM_LAYERS = 28;
    constexpr int NUM_HEADS = 2;
    constexpr int HEAD_DIM = 128;
    std::vector<int64_t> past_shape = {batch_size, NUM_HEADS, 0, HEAD_DIM};
    std::vector<Ort::Float16_t> empty_data;

    std::vector<Ort::Value> inputs;
    inputs.reserve(3 + 2 * NUM_LAYERS);
    inputs.push_back(std::move(input_ids_tensor));
    inputs.push_back(std::move(attn_mask_tensor));
    inputs.push_back(std::move(pos_ids_tensor));

    for (int layer = 0; layer < NUM_LAYERS; ++layer) {
        auto key_tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
            mem_info, empty_data.data(), empty_data.size(),
            past_shape.data(), past_shape.size());
        auto value_tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
            mem_info, empty_data.data(), empty_data.size(),
            past_shape.data(), past_shape.size());
        inputs.push_back(std::move(key_tensor));
        inputs.push_back(std::move(value_tensor));
    }

    // 准备输入/输出名称指针
    std::vector<const char*> input_names_ptr;
    for (const auto& name : input_names) input_names_ptr.push_back(name.c_str());

    std::vector<const char*> output_names_ptr;
    for (const auto& name : output_names) output_names_ptr.push_back(name.c_str());

    try {
        // 第一次推理（整个 prompt）
        auto start_time = std::chrono::high_resolution_clock::now();
        auto outputs = ctx.session->Run(Ort::RunOptions{nullptr},
                                        input_names_ptr.data(), inputs.data(), inputs.size(),
                                        output_names_ptr.data(), output_names_ptr.size());
        auto end_time = std::chrono::high_resolution_clock::now();
        std::cout << "first inference time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()
                  << " ms" << std::endl;

        if (outputs.empty()) return;

        // 处理第一次的输出：logits + present KV
        auto& logits = outputs[0];
        auto shape = logits.GetTensorTypeAndShapeInfo().GetShape();
        auto* logits_data = logits.GetTensorMutableData<float>();
        std::cout << "logits shape: ";
        for (auto dim : shape) std::cout << dim << " ";
        std::cout << std::endl;

        // 提取 present KV（用于后续步骤）
        std::vector<Ort::Value> past_kv;
        past_kv.reserve(outputs.size() - 1);
        for (size_t i = 1; i < outputs.size(); ++i) {
            past_kv.push_back(std::move(outputs[i]));
        }

        // 采样第一个新 token（使用 argmax）
        int64_t batch_idx = 0;
        int64_t last_pos = shape[1] - 1;
        int64_t vocab_size = shape[2];
        float* last_token_logits = logits_data + (batch_idx * shape[1] + last_pos) * vocab_size;
        int64_t next_token_id = std::distance(
            last_token_logits,
            std::max_element(last_token_logits, last_token_logits + vocab_size)
        );
        generated_tokens.push_back(next_token_id);
        std::cout << "step 0 next_token_id: " << next_token_id << std::endl;

        // 自回归生成循环
        for (int step = 1; step < max_new_tokens; ++step) {
            // 准备当前 step 的输入：只包含新生成的 token
            std::vector<int64_t> step_input_ids = {next_token_id};
            std::vector<int64_t> step_attention_mask = {1};                 // 长度为 1
            std::vector<int64_t> step_position_ids = {current_seq_len};    // 当前 token 的绝对位置
            std::vector<int64_t> step_shape = {batch_size, 1};              // [1, 1]

            auto step_input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
                mem_info, step_input_ids.data(), step_input_ids.size(),
                step_shape.data(), step_shape.size());

            auto step_attn_mask_tensor = Ort::Value::CreateTensor<int64_t>(
                mem_info, step_attention_mask.data(), step_attention_mask.size(),
                step_shape.data(), step_shape.size());

            auto step_pos_ids_tensor = Ort::Value::CreateTensor<int64_t>(
                mem_info, step_position_ids.data(), step_position_ids.size(),
                step_shape.data(), step_shape.size());

            // 构建 step inputs：前三个静态输入 + past_kv
            std::vector<Ort::Value> step_inputs;
            step_inputs.reserve(3 + past_kv.size());
            step_inputs.push_back(std::move(step_input_ids_tensor));
            step_inputs.push_back(std::move(step_attn_mask_tensor));
            step_inputs.push_back(std::move(step_pos_ids_tensor));
            for (auto& kv : past_kv) {
                step_inputs.push_back(std::move(kv));   // 移动 past_kv 中的值
            }
            // past_kv 已被移动，此后不应再使用

            auto step_start = std::chrono::high_resolution_clock::now();
            auto step_outputs = ctx.session->Run(Ort::RunOptions{nullptr},
                                                 input_names_ptr.data(), step_inputs.data(), step_inputs.size(),
                                                 output_names_ptr.data(), output_names_ptr.size());
            auto step_end = std::chrono::high_resolution_clock::now();
            std::cout << "step " << step << " inference time: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(step_end - step_start).count()
                      << " ms" << std::endl;

            if (step_outputs.empty()) break;

            // 解析 step 输出
            auto& step_logits = step_outputs[0];
            auto step_shape_info = step_logits.GetTensorTypeAndShapeInfo();
            auto step_shape_dims = step_shape_info.GetShape();
            auto* step_logits_data = step_logits.GetTensorMutableData<float>();
            int64_t step_vocab_size = step_shape_dims[2];
            // 因为输入长度为 1，所以最后一个位置就是第 0 个位置
            float* step_last_logits = step_logits_data;
            int64_t step_next_token = std::distance(
                step_last_logits,
                std::max_element(step_last_logits, step_last_logits + step_vocab_size)
            );

            generated_tokens.push_back(step_next_token);
            std::cout << "step " << step << " next_token_id: " << step_next_token << std::endl;

            // 提取新的 present KV 作为下一轮的 past KV
            std::vector<Ort::Value> next_past_kv;
            next_past_kv.reserve(step_outputs.size() - 1);
            for (size_t i = 1; i < step_outputs.size(); ++i) {
                next_past_kv.push_back(std::move(step_outputs[i]));
            }
            past_kv = std::move(next_past_kv);

            // 更新总序列长度，供下一次 position_id 使用
            ++current_seq_len;
            next_token_id = step_next_token;

            // 可选的停止条件：遇到 EOS token（例如 151647）
            if (step_next_token == 151647) {
                std::cout << "EOS token reached, stopping generation." << std::endl;
                break;
            }
        }

        // 打印所有生成的 token ids
        std::cout << "generated token ids: ";
        for (const auto& tid : generated_tokens) std::cout << tid << ' ';
        std::cout << std::endl;

    } catch (const Ort::Exception& e) {
        std::cerr << "ORT inference failed: " << e.what() << std::endl;
    }
}


void gdlz::ort::core::OnnxRtFramework::run_batch_init(void* tokens, int len, data::OnnxRTEngine& ctx,
    data::OnnxExamKV& kv) const
{
    auto* intPtr = static_cast<int64_t*>(tokens);
    std::vector<int64_t> input_ids(intPtr, intPtr + len);

    std::cout<<"input_ids: ";
    for (auto input_id : input_ids){
        std::cout<<input_id<<" ";
    }
    std::cout<<std::endl;


    if (!ctx.session) return;

    Ort::AllocatorWithDefaultOptions allocator;
    const size_t numInputs =  ctx.session->GetInputCount();
    const size_t numOutputs = ctx.session->GetOutputCount();

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

    for (size_t i = 0; i != numInputs; ++i) {
        auto input_name_ptr = ctx.session->GetInputNameAllocated(i, allocator);
        input_names.emplace_back(input_name_ptr.get());
    }

    for (size_t i = 0; i != numOutputs; ++i) {
        auto output_name_ptr = ctx.session->GetOutputNameAllocated(i, allocator);
        output_names.emplace_back(output_name_ptr.get());
    }


    auto current_seq_len = static_cast<int64_t>(input_ids.size());
    // 创建第一次推理的输入张量
    constexpr int64_t batch_size = 1;

    // 第一次推理所需的 attention_mask 和 position_ids（完整序列）
    std::vector<int64_t> attention_mask(current_seq_len, 1);
    std::vector<int64_t> position_ids(current_seq_len);
    for (int64_t i = 0; i < current_seq_len; ++i) position_ids[i] = i;


    // 使用 CPU 内存（可根据需要改为 GPU）
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector input_shape = {batch_size, current_seq_len};

    auto input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
        mem_info, input_ids.data(), input_ids.size(),
        input_shape.data(), input_shape.size());

    auto attn_mask_tensor = Ort::Value::CreateTensor<int64_t>(
        mem_info, attention_mask.data(), attention_mask.size(),
        input_shape.data(), input_shape.size());

    auto pos_ids_tensor = Ort::Value::CreateTensor<int64_t>(
        mem_info, position_ids.data(), position_ids.size(),
        input_shape.data(), input_shape.size());


    const int NUM_LAYERS = 28;
    constexpr int NUM_HEADS = 2;
    constexpr int HEAD_DIM = 128;
    std::vector<int64_t> past_shape = {batch_size, NUM_HEADS, 0, HEAD_DIM};
    std::vector<Ort::Float16_t> empty_data;

    std::vector<Ort::Value> inputs;
    inputs.reserve(3 + 2 * NUM_LAYERS);
    inputs.push_back(std::move(input_ids_tensor));
    inputs.push_back(std::move(attn_mask_tensor));
    inputs.push_back(std::move(pos_ids_tensor));

    for (int layer = 0; layer < NUM_LAYERS; ++layer) {
        auto key_tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
            mem_info, empty_data.data(), empty_data.size(),
            past_shape.data(), past_shape.size());

        auto value_tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
            mem_info, empty_data.data(), empty_data.size(),
            past_shape.data(), past_shape.size());
        inputs.push_back(std::move(key_tensor));
        inputs.push_back(std::move(value_tensor));
    }


    std::vector<const char*> input_names_ptr;
    for (const auto& name : input_names) input_names_ptr.push_back(name.c_str());

    std::vector<const char*> output_names_ptr;
    for (const auto& name : output_names) output_names_ptr.push_back(name.c_str());

    auto outputs = ctx.session->Run(
           Ort::RunOptions{nullptr},
           input_names_ptr.data(), inputs.data(), inputs.size(),
            output_names_ptr.data(), output_names_ptr.size()
       );
    auto& logits = outputs[0];
    auto shape = logits.GetTensorTypeAndShapeInfo().GetShape();
    auto* logits_data = logits.GetTensorData<float>();
    int64_t batch_idx = 0;
    int64_t last_pos = shape[1] - 1;
    int64_t vocab_size = shape[2];

    const float* last_token_logits = logits_data + (batch_idx * shape[1] + last_pos) * vocab_size;
    int64_t next_token_id = std::distance(
        last_token_logits,
        std::max_element(last_token_logits, last_token_logits + vocab_size)
    );

    kv.next_token_id = next_token_id;
    std::cout<<"next_token_id: "<<next_token_id<<std::endl;
    kv.token_count = len;
    std::vector<Ort::Value> next_past_kv;
    next_past_kv.reserve(outputs.size() - 1);
    for (size_t i = 1; i < outputs.size(); ++i) {
        next_past_kv.push_back(std::move(outputs[i]));
    }

    kv.next_past_kv = std::move(next_past_kv);
}

void gdlz::ort::core::OnnxRtFramework::run_token_one_step( data::OnnxRTEngine& ctx,
                                                           data::OnnxExamKV& kv) const
{
    kv.token_count++;
    std::vector step_input_ids = {kv.next_token_id};
    std::vector<int64_t> step_attention_mask = {1};                 // 长度为 1
    std::vector<int64_t> step_position_ids = {kv.token_count-1};    // 当前 token 的绝对位置
    std::vector<int64_t> step_shape = {1, 1};              // [1, 1]

    Ort::AllocatorWithDefaultOptions allocator;
    const size_t numInputs = ctx.session->GetInputCount();
    const size_t numOutputs = ctx.session->GetOutputCount();

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

    for (size_t i = 0; i != numInputs; ++i) {
        auto input_name_ptr = ctx.session->GetInputNameAllocated(i, allocator);
        input_names.emplace_back(input_name_ptr.get());
    }

    for (size_t i = 0; i != numOutputs; ++i) {
        auto output_name_ptr = ctx.session->GetOutputNameAllocated(i, allocator);
        output_names.emplace_back(output_name_ptr.get());
    }

    std::vector<const char*> input_names_ptr;
    for (const auto& name : input_names) input_names_ptr.push_back(name.c_str());

    std::vector<const char*> output_names_ptr;
    for (const auto& name : output_names) output_names_ptr.push_back(name.c_str());

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    auto step_input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
                mem_info, step_input_ids.data(), step_input_ids.size(),
                step_shape.data(), step_shape.size());

    auto step_attn_mask_tensor = Ort::Value::CreateTensor<int64_t>(
        mem_info, step_attention_mask.data(), step_attention_mask.size(),
        step_shape.data(), step_shape.size());

    auto step_pos_ids_tensor = Ort::Value::CreateTensor<int64_t>(
        mem_info, step_position_ids.data(), step_position_ids.size(),
        step_shape.data(), step_shape.size());

    std::vector<Ort::Value> step_inputs;
    step_inputs.reserve(3 + kv.next_past_kv.size());
    step_inputs.push_back(std::move(step_input_ids_tensor));
    step_inputs.push_back(std::move(step_attn_mask_tensor));
    step_inputs.push_back(std::move(step_pos_ids_tensor));
    for (auto& kv_1 : kv.next_past_kv) {
        step_inputs.push_back(std::move(kv_1));   // 移动 past_kv 中的值
    }
    auto step_outputs = ctx.session->Run(Ort::RunOptions{nullptr},
                                                input_names_ptr.data(), step_inputs.data(), step_inputs.size(),
                                                output_names_ptr.data(), output_names_ptr.size()
    );
    auto& step_logits = step_outputs[0];
    auto step_shape_info = step_logits.GetTensorTypeAndShapeInfo();
    auto step_shape_dims = step_shape_info.GetShape();
    auto* step_logits_data = step_logits.GetTensorMutableData<float>();

    int64_t step_vocab_size = step_shape_dims[2];

    // 因为输入长度为 1，所以最后一个位置就是第 0 个位置
    float* step_last_logits = step_logits_data;
    int64_t step_next_token = std::distance(
        step_last_logits,
        std::max_element(step_last_logits, step_last_logits + step_vocab_size)
    );

    kv.next_token_id = step_next_token;
    std::vector<Ort::Value> next_past_kv;
    next_past_kv.reserve(step_outputs.size() - 1);
    for (size_t i = 1; i < step_outputs.size(); ++i) {
        next_past_kv.push_back(std::move(step_outputs[i]));
    }
    kv.next_past_kv = std::move(next_past_kv);
    std::cout<<"step next token id: "<<step_next_token<<std::endl;
}
