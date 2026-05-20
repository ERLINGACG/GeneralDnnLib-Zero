//
// Created by HP on 2026/5/3.
//

#ifndef GENERALDNNLIB_ZERO_ONNX_RT_FRAMEWORK_H
#define GENERALDNNLIB_ZERO_ONNX_RT_FRAMEWORK_H
#include "gdlz/ort/data/onnx_rt_data.h"
#include <iostream>
#include "gdlz/export.h"
namespace gdlz::ort::core{

     class  GDLZ_CPP_API  OnnxRtFramework
     {
         data::OnnxRTEngineInfo engine_info;
     public:
         OnnxRtFramework() = default;
         ~OnnxRtFramework() = default;

         OnnxRtFramework* ResourceDirector(const char* config_path);

         int            CreateEngine(data::OnnxRTEngine& engine) const;
         static int     CreateEngine(const data::OnnxRTEngineInfo& engine_info,data::OnnxRTEngine& engine);
         static void    GetEngineInfo(data::OnnxRTEngine& ctx);

         void   run(data::OnnxRTEngine& ctx) const;
         void   run_batch_init(void* tokens,int len,data::OnnxRTEngine& ctx,data::OnnxExamKV& kv) const;
         void   run_token_one_step(data::OnnxRTEngine& ctx,data::OnnxExamKV& kv)  const;


         data::OnnxRTEngineInfo& GetEngineInfo_() {
             return engine_info;
         }


         template <typename T>
         static int SetInput(const T& engine, const OrtMemoryInfo* mem_info,
                            const char* input_name,
                            void* shape, int64_t shape_len,
                            void* data,  int64_t data_len,
                            int32_t data_type,
                            data::OnnxRtInput& input) {

                 // 1. 获取允许的输入名称列表
                 auto& input_names_ptr = engine.get_core_engine().input_name_ptr;
                 if (!input_names_ptr) return -1;  // 空指针检查

                 const auto& input_names = *input_names_ptr;  // 解引用得到 vector<string>

                 bool name_valid = (std::string(input_name) == "AUTO") ||
                                   (std::find(input_names.begin(), input_names.end(), input_name) != input_names.end());

                 if (!name_valid) return -1;

                // 2. 构建形状 vector (int64)
                auto* shape_ptr = static_cast<int64_t*>(shape);
                std::vector<int64_t> shape_vec(shape_ptr, shape_ptr + shape_len);

                // 3. 根据数据类型创建 Ort::Value 并移动进容器
                switch (data_type) {
                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
                        auto* data_ptr = static_cast<int64_t*>(data);
                        auto tensor = Ort::Value::CreateTensor<int64_t>(
                            mem_info, data_ptr, data_len,
                            shape_vec.data(), shape_vec.size());
                        input.input_ptr->push_back(std::move(tensor));
                        break;
                    }
                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
                        auto* data_ptr = static_cast<float*>(data);
                        auto tensor = Ort::Value::CreateTensor<float>(
                            mem_info, data_ptr, data_len,
                            shape_vec.data(), shape_vec.size());
                        input.input_ptr->push_back(std::move(tensor));
                        break;
                    }

                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
                        // 注意：ONNX 的 float16 通常用 uint16_t 或专用类型存储
                        auto* data_ptr = static_cast<Ort::Float16_t*>(data);
                        auto tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
                            mem_info, data_ptr, data_len,
                            shape_vec.data(), shape_vec.size());
                        // 可能需要额外指定数据类型的 OrtMemType 等，这里按默认
                        input.input_ptr->push_back(std::move(tensor));
                        break;
                    }

                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
                        auto* data_ptr = static_cast<int32_t*>(data);
                        auto tensor = Ort::Value::CreateTensor<int32_t>(
                            mem_info, data_ptr, data_len,
                            shape_vec.data(), shape_vec.size());
                        input.input_ptr->push_back(std::move(tensor));
                        break;
                    }

                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {
                        auto* data_ptr = static_cast<int8_t*>(data);
                        auto tensor = Ort::Value::CreateTensor<int8_t>(
                            mem_info, data_ptr, data_len,
                            shape_vec.data(), shape_vec.size());
                        input.input_ptr->push_back(std::move(tensor));
                        break;
                    }

                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
                        auto* data_ptr = static_cast<uint8_t*>(data);
                        auto tensor = Ort::Value::CreateTensor<uint8_t>(
                            mem_info, data_ptr, data_len,
                            shape_vec.data(), shape_vec.size());
                        input.input_ptr->push_back(std::move(tensor));
                        break;
                    }

                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
                        auto* data_ptr = static_cast<bool*>(data);
                        auto tensor = Ort::Value::CreateTensor<bool>(
                            mem_info, data_ptr, data_len,
                            shape_vec.data(), shape_vec.size());
                        input.input_ptr->push_back(std::move(tensor));
                        break;
                    }
                    // 根据需要继续添加 DOUBLE, INT16, UINT16, UINT32, UINT64 等
                    default:
                        return -2;  // 不支持的数据类型
                }
                return 0;
        }


        template <typename T,typename I>
        static int SetOutput(const T& engine, const OrtMemoryInfo* mem_info,
                            const char* output_name,
                            void* shape, int64_t shape_len,
                            void* data,  int64_t data_len,
                            int32_t data_type,
                            I& input_i ) {

                 // 1. 获取允许的输出名称列表
                 auto& output_names_ptr = engine.get_core_engine().output_name_ptr;
                 auto& input = input_i.get_input();
                 if (!output_names_ptr) return -1;  // 空指针检查

                 const auto& output_names = *output_names_ptr;  // 解引用得到 vector<string>

                 bool name_valid = (std::string(output_name) == "AUTO") ||
                                   (std::find(output_names.begin(), output_names.end(), output_name) != output_names.end());

                 if (!name_valid) return -1;

                // 2. 构建形状 vector (int64)
                auto* shape_ptr = static_cast<int64_t*>(shape);
                std::vector<int64_t> shape_vec(shape_ptr, shape_ptr + shape_len);

                // 3. 根据数据类型创建 Ort::Value 并移动进容器
                switch (data_type) {
                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
                        auto* data_ptr = static_cast<int64_t*>(data);
                        auto tensor = Ort::Value::CreateTensor<int64_t>(
                            mem_info, data_ptr, data_len,
                            shape_vec.data(), shape_vec.size());
                        input.input_ptr->push_back(std::move(tensor));
                        break;
                    }
                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
                        auto* data_ptr = static_cast<float*>(data);
                        auto tensor = Ort::Value::CreateTensor<float>(
                            mem_info, data_ptr, data_len,
                            shape_vec.data(), shape_vec.size());
                        input.input_ptr->push_back(std::move(tensor));
                        break;
                    }
                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
                        // 注意：ONNX 的 float16 通常用 uint16_t 或专用类型存储
                        auto* data_ptr = static_cast<uint16_t*>(data);
                        auto tensor = Ort::Value::CreateTensor<uint16_t>(
                            mem_info, data_ptr, data_len,
                            shape_vec.data(), shape_vec.size());
                        // 可能需要额外指定数据类型的 OrtMemType 等，这里按默认
                        input.input_ptr->push_back(std::move(tensor));
                        break;
                    }

                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
                        auto* data_ptr = static_cast<int32_t*>(data);
                        auto tensor = Ort::Value::CreateTensor<int32_t>(
                            mem_info, data_ptr, data_len,
                            shape_vec.data(), shape_vec.size());
                        input.input_ptr->push_back(std::move(tensor));
                        break;
                    }

                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {
                        auto* data_ptr = static_cast<int8_t*>(data);
                        auto tensor = Ort::Value::CreateTensor<int8_t>(
                            mem_info, data_ptr, data_len,
                            shape_vec.data(), shape_vec.size());
                        input.input_ptr->push_back(std::move(tensor));
                        break;
                    }

                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
                        auto* data_ptr = static_cast<uint8_t*>(data);
                        auto tensor = Ort::Value::CreateTensor<uint8_t>(
                            mem_info, data_ptr, data_len,
                            shape_vec.data(), shape_vec.size());
                        input.input_ptr->push_back(std::move(tensor));
                        break;
                    }

                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
                        auto* data_ptr = static_cast<bool*>(data);
                        auto tensor = Ort::Value::CreateTensor<bool>(
                            mem_info, data_ptr, data_len,
                            shape_vec.data(), shape_vec.size());
                        input.input_ptr->push_back(std::move(tensor));
                        break;
                    }
                    // 根据需要继续添加 DOUBLE, INT16, UINT16, UINT32, UINT64 等
                    default:
                        return -2;  // 不支持的数据类型
                }
                return 0;
        }
     };
}
#endif //GENERALDNNLIB_ZERO_ONNX_RT_FRAMEWORK_H