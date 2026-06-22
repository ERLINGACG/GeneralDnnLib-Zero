//
// Created by HP on 2026/5/22.
//

#include "gdlz/ort/emd/onnx_rt_emd_framework.h"

#include "gdlz/ort/core/onnx_rt_core_director.h"
#include "gdlz/ort/core/onnx_rt_framework.h"
#include "gdlz/ort/emd/onnx_rt_emd_data.h"
#include "gdlz/ort/emd/onnx_rt_emd_director.h"
#include "nlohmann/json.hpp"

#include <algorithm>
gdlz::ort::emd::OnnxRtEmdFramework::OnnxRtEmdFramework(const char* config_path){
    nlohmann::json j;
    OnnxRtEmdDirector::Handle(engine_info,config_path, j);
    OnnxRtEmdDirector::CreateEngine(engine, engine_info, j);
}

gdlz::ort::emd::OnnxRtEmdFramework::~OnnxRtEmdFramework()
{
    ResetEngine();
}

int gdlz::ort::emd::OnnxRtEmdFramework::GetEngineInfo()
{
    core::OnnxRtFramework::GetEngineInfo(engine.core_engine);
    return 0;
}

int gdlz::ort::emd::OnnxRtEmdFramework::SetShape(core::shape::OnnxRtShape& shape, data::OnnxRtEmdCtx& ctx)
{
    try
    {
        auto info=[&]()
        {
            std::cout<<"======SetShape======"<<std::endl;
            for (auto& item : *ctx.shape){
                std::cout<<"name:"<<item.name<<std::endl;
                std::cout<<"shape_type:"<<item.shape_type<<std::endl;
                std::cout<<"data_type:"<<item.data_type<<std::endl;
                std::cout<<"data_len:"<<item.data_len<<std::endl;
                std::cout<<"dim:[";
                for (int i=0;i<item.dim;i++){
                    std::cout<<item.shape[i]<<" ";
                }
                std::cout<<"]"<<std::endl;
            }
        };

        auto init=[&](){
            if (ctx.shape==nullptr){
                ctx.shape=std::make_unique<std::vector<core::shape::OnnxRtShape>>();
            }
        };

        init();
        ctx.shape->push_back(shape);
        // info();

        return 0;
    }catch (std::exception& e){
        return -1;
    }
}

int gdlz::ort::emd::OnnxRtEmdFramework::GetAvgPoolEmd(data::OnnxRtEmdData& data, std::vector<Ort::Value>& emd){
    try{
        if (emd.empty()) {
            return -1;
        }

        auto& output = emd[0];
        if (!output.IsTensor()) {
            return -1;
        }

        const auto shape = output.GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() != 3) {
            return -1;
        }

        const auto batch_size = shape[0];
        const auto sequence_length = shape[1];
        const auto hidden_size = shape[2];
        if (batch_size <= 0 || sequence_length <= 0 || hidden_size <= 0) {
            return -1;
        }

        const auto* tensor_data = output.GetTensorData<float>();
        if (tensor_data == nullptr) {
            return -1;
        }

        const auto pooled_size = static_cast<size_t>(batch_size) * static_cast<size_t>(hidden_size);
        std::vector<float> pooled(pooled_size, 0.0f);

        for (int64_t batch_index = 0; batch_index < batch_size; ++batch_index) {
            for (int64_t hidden_index = 0; hidden_index < hidden_size; ++hidden_index) {
                float sum = 0.0f;
                for (int64_t seq_index = 0; seq_index < sequence_length; ++seq_index) {
                    const auto offset =
                        (static_cast<size_t>(batch_index) * static_cast<size_t>(sequence_length) +
                         static_cast<size_t>(seq_index)) * static_cast<size_t>(hidden_size) +
                        static_cast<size_t>(hidden_index);
                    sum += tensor_data[offset];
                }

                const auto pooled_offset =
                    static_cast<size_t>(batch_index) * static_cast<size_t>(hidden_size) +
                    static_cast<size_t>(hidden_index);
                pooled[pooled_offset] = sum / static_cast<float>(sequence_length);
            }
        }

        if (data.embedding == nullptr) {
            data.embedding = new float[pooled_size];
        }
        std::ranges::copy(pooled, data.embedding);
        data.embedding_size = static_cast<int>(pooled_size);
        return 0;
    }
    catch (const std::exception&)
    {
        return -1;
    }
}

int gdlz::ort::emd::OnnxRtEmdFramework::Forward(data::OnnxRtEmdCtx& ctx, data::OnnxRtEmdData& data) const
{
    if (ctx.shape==nullptr){
        return -1;
    }

    std::vector<char*> input_names;
    std::vector<char*> output_names;

    std::vector<Ort::Value> inputs;
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto build_inputs=[&](){
        for (auto& shape : *ctx.shape){
            if (shape.shape_type==0){
                input_names.push_back(shape.name);
            }else{
                output_names.push_back(shape.name);
            }
            if (shape.data_type==ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64){
                const auto data_array=static_cast<int64_t*>(shape.data);
                std::vector<int64_t> input_data={data_array,data_array+shape.data_len};
                auto tensor=Ort::Value::CreateTensor<int64_t>(
                    mem_info,
                    input_data.data(),
                    input_data.size(),
                    shape.shape,
                    shape.dim
                );
                inputs.push_back(std::move(tensor));
            }
        }

    };

    std::vector<Ort::Value> outputs;
    auto build_outputs=[&]()
    {
          outputs=engine.core_engine.session->Run(
              Ort::RunOptions{},
              input_names.data(),
              inputs.data(),
              inputs.size(),
              output_names.data(),
              output_names.size()
          );
    };

    auto get_output=[&](){
        return GetAvgPoolEmd(data, outputs);
    };

    build_inputs();
    build_outputs();
    return get_output();
}

int gdlz::ort::emd::OnnxRtEmdFramework::ResetEngine()
{
    this->engine.Reset();
    return 0;
}
