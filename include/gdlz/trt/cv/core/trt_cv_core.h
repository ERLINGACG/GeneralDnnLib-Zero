//
// Created by HP on 2026/7/8.
//

#ifndef GENERALDNNLIB_ZERO_TRT_CV_CORE_H
#define GENERALDNNLIB_ZERO_TRT_CV_CORE_H
#include "trt_cv_core_data.h"
#include "gdlz/export.h"
namespace gdlz::trt::icv::core{
    class GDLZ_CPP_API TensorRTCvCoreFramework{
        TensorRTCvEngine engine;
        public:
            TensorRTCvCoreFramework(const char* config_path);
            ~TensorRTCvCoreFramework()=default;

            int CreateEngine();
            int CreateCtx(TensorRTCvCtx& ctx);
            int GetEngineInfo();
            static int RegisterInputLayer (TensorRTCvCtx& ctx,const char* name,const int64_t* shape,int64_t shape_len);
            static int RegisterOutputLayer(TensorRTCvCtx& ctx,const char* name,trt::core::cuda::DataType dataType);
            static int FreeLayer(TensorRTCvCtx& ctx,const char* name);

            static int PostImg(const char* name,TensorRTCvCtx& ctx,
                               void* img, int64_t len,
                               trt::core::cuda::DataType dataType);

            static int Forward(TensorRTCvCtx& ctx);
            static int GetOutput(const char* name,TensorRTCvCtx& ctx,
                                 trt::core::cuda::TensorRTCudaData& data,
                                 trt::core::cuda::DataType dataType);


            template<typename T, typename... Args>
            static int PreProcess(T& processer,Args&&... args){
                return processer.PreProcess(std::forward<Args>(args)...);
            }

            template<typename T, typename... Args>
            static int PostProcess(T& processer, float* output, Args&&... args){
                return processer.PostProcess(output, std::forward<Args>(args)...);
            }
    };
}
#endif //GENERALDNNLIB_ZERO_TRT_CV_CORE_H