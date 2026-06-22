//
// Created by HP on 2026/6/2.
//

#ifndef GENERALDNNLIB_ZERO_ONNX_RT_CODE_SHAPE_H
#define GENERALDNNLIB_ZERO_ONNX_RT_CODE_SHAPE_H
#include <cstdint>
#include <vector>

namespace gdlz::ort::core::shape{

    struct OnnxRtShape{
        char*     name;
        void*     data;
        int64_t*  shape;
        int32_t   dim;
        int32_t   data_len;
        int32_t   data_type;
        int32_t   shape_type;


        template<typename T>
        int ReadData(std::vector<T>& data_vec){
            data_vec.resize(data_len);
            memcpy(data_vec.data(), data, data_len * sizeof(T));
            return 0;
        }

    };

}
#endif //GENERALDNNLIB_ZERO_ONNX_RT_CODE_SHAPE_H