//
// Created by HP on 2026/5/15.
//

#include "gdlz/ort/core/onnx_rt_core_director.h"
#include "gdlz/ort/core/onnx_rt_framework.h"
#include "gdlz/ort/llm/onnx_rt_llm_director.h"

#include <iostream>
#include <dylog/logger.h>

void gdlz::ort::llm::OnnxRtLLmDirector::Handle(
    ort::data::OnnxRTEngineInfo& engine_info,
    const char* config_path,nlohmann::json& config
) {
    core::OnnxRtCoreDirector::Handle(engine_info, config_path, config);
}

int gdlz::ort::llm::OnnxRtLLmDirector::CreateEngine(
    data::OnnxRtLLmEngine& engine,
    const ort::data::OnnxRTEngineInfo& engine_info,
    nlohmann::json& config
) {
    try
    {
        engine.head_dim  =  config["head_dim"].get<int>();
        engine.layers    =  config["layers"].get<int>();
        engine.heads     =  config["heads"].get<int>();


        for (auto& shape: config["struct"]) {

            // if (shape["name"]=="input_ids" || shape["name"]=="attention_mask" || shape["name"]=="position_ids") {
            //     engine.struct_info.push_back(std::map<std::string,std::vector<std::string>>{
            //         {shape["name"].get<std::string>(),
            //             shape["shape"].get<std::vector<std::string>>()}
            //         }
            //     );
            // }
            // if (shape["name"]=="kv_shape_key" || shape["name"]=="kv_shape_value") {
            //     engine.kv_shape.push_back(data::KvShape{
            //                shape["type"].get<std::string>(),
            //               shape["shape"].get<std::vector<int64_t>>(),
            //               shape["num"].get<int32_t>(),
            //     });
            // }
            engine.input_shapes.push_back(data::InputShape{
                shape["type"].get<std::string>(),
                shape["data_type"].get<std::string>(),
                shape["name"].get<std::string>(),
                shape["layer_pos"].get<int32_t>(),
                shape["num"].get<int32_t>(),
                shape["shape"].get<std::vector<int64_t>>(),
            });
        }

        std::cout<<"input_shapes size "<<engine.input_shapes.size()<<std::endl;

        auto test_layer=std::vector<std::string>(engine.layers);
        for (auto& shape: engine.input_shapes) {
            std::cout<<"shape "<<shape.name<<" | "<<shape.layer_pos<<" | "<<shape.num<<" | "<<shape.shape.size()<<std::endl;
            for (int i=0; i<shape.num; i++) {
                    test_layer[shape.layer_pos+i]= shape.type;

            }
        }
        std::cout<<"test_layer size "<<test_layer.size()<<std::endl;
        for (auto& layer: test_layer) {
            std::cout<<layer<<std::endl;
        }

        core::OnnxRtFramework::CreateEngine(engine_info, engine.core_engine);
        return 0;
    }catch (std::exception& e) {
        std::cout<<"CreateEngine error "<<e.what()<<std::endl;
        return -1;
    }
}
