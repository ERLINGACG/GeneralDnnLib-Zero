//
// Created by HP on 2026/4/19.
//

#ifndef GENERALDNNLIB_ZERO_TRT_EMB_FRAMEWORK_H
#define GENERALDNNLIB_ZERO_TRT_EMB_FRAMEWORK_H
#include <cuda_fp16.h>
#include <unordered_map>
#include <vector>

#include "NvInfer.h"
#include "trt_emb_data.h"
#include "trt_emb_director.h"
#define CHECK_CUDA(status) \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(status) << std::endl; \
        return -1; \
    }

namespace gdlz::trt::rag{

    class TrtEmbFramework {

        TrtEmbEngine engine;
        const std::unordered_map<nvinfer1::DataType, size_t> typeSizeMap = {
            {nvinfer1::DataType::kINT32, sizeof(int32_t)},
            {nvinfer1::DataType::kINT64, sizeof(int64_t)},
            {nvinfer1::DataType::kFLOAT, sizeof(float)},
            {nvinfer1::DataType::kHALF, sizeof(half)},
            {nvinfer1::DataType::kINT8, sizeof(int8_t)},
            {nvinfer1::DataType::kUINT8, sizeof(uint8_t)},
        };

        public:
            TrtEmbFramework() = default;
            ~TrtEmbFramework() = default;



            void  ResourceDirector(
                TrtEmbDirector& director,
                const char* config_path
            );
            void GetEngineInfo() const;
            int CreateCtx(TrtEmbCtx& ctx) const;

            int PostCuda(const TrtEmbCtx& ctx,
                         const char* name,
                         const void* input,
                         int len,
                         int isMask
            ) const;

            static int SetDims(
                    TrtEmbCtx& ctx,
                    const char* name,
                    const int* input, int len
            );

            int GetDims(TrtEmbCtx& ctx,const char* name) const;
            int Forward(TrtEmbCtx& ctx) const;
            int GetEmbedding(const TrtEmbCtx& ctx,
                            TrtEmbData& embedding,
                            const char* name
            ) const;




            template<typename  F>
            TrtEmbFramework* load(F f,const char* config_path){
                   auto director = f();
                   ResourceDirector(director,config_path);
                    return this;
            };
    };

}
#endif //GENERALDNNLIB_ZERO_TRT_EMB_FRAMEWORK_H