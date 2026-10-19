//
// Created by HP on 2026/7/8.
//

#ifndef GENERALDNNLIB_ZERO_TRT_CV_CORE_H
#define GENERALDNNLIB_ZERO_TRT_CV_CORE_H
// #include <complex.h>

#include "trt_cv_core_data.h"
#include "gdlz/export.h"
#include "gdlz/trt/core/trt_core_data.h"
#include "gdlz/trt/core/cuda/trt_cuda_data.h"
#include "gdlz/trt/core/cuda/buffer/trt_cuda_buffer.h"
#include "gdlz/trt/core/cuda/buffer/trt_cuda_buffer_type.h"
namespace gdlz::trt::icv::core{
    class GDLZ_CPP_API TensorRTCvCoreFramework{
        TensorRTCvEngine engine;
        public:
            TensorRTCvCoreFramework(const char* config_path);
            ~TensorRTCvCoreFramework()=default;

            int CreateEngine();
            int CreateCtx(TensorRTCvCtx& ctx);
            int GetEngineInfo();

            // static int RegisterInputLayer (TensorRTCvCtx& ctx,const char* name,const int64_t* shape,int64_t shape_len);
            // static int RegisterOutputLayer(TensorRTCvCtx& ctx,const char* name,trt::core::cuda::DataType dataType);

            static int ResetInputLayer    (TensorRTCvCtx& ctx,const char* name,const int64_t* shape,int64_t shape_len);
            static int ResetInputLayer    (TensorRTCvCtx& ctx,const char* name,const int64_t* shape,int64_t shape_len,trt::core::cuda::BufferType);
            // static int ResetOutputLayer   (TensorRTCvCtx& ctx,const char* name,trt::core::cuda::DataType dataType);
            static int ResetOutputLayer   (TensorRTCvCtx& ctx,const char* name,trt::core::cuda::BufferType bufferType);


            static void UploadData(const char* name,TensorRTCvCtx& ctx,const std::vector<std::byte>& host);

            static void DownloadData(const char* name,TensorRTCvCtx& ctx);

            static int BindAddress(const char* name,TensorRTCvCtx& ctx,void* address);

            static const void* GetLayerAddress(const char* name, TensorRTCvCtx& ctx) ;

            static int Forward(TensorRTCvCtx& ctx);

            static int64_t* GetOutputShapes(const char* name, TensorRTCvCtx& ctx);

            static int64_t  GetOutputShape(const char* name, TensorRTCvCtx& ctx);


            static trt::core::cuda::TensorRTCudaBuffer& GetBuffer(const char* name, TensorRTCvCtx& ctx);

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