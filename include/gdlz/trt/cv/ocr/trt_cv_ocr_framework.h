//
// Created by HP on 2026/7/2.
//

#ifndef GENERALDNNLIB_ZERO_TRT_CV_OCR_FRAMEWORK_H
#define GENERALDNNLIB_ZERO_TRT_CV_OCR_FRAMEWORK_H
#include "trt_cv_ocr_data.h"

// #include "../../../../../../Clib/opencv/include/opencv2/core/mat.hpp"
#include <opencv2/opencv.hpp>
#include "gdlz/export.h"
namespace gdlz::trt::icv::ocr {
    class GDLZ_CPP_API TensorRTCvOCRFramework{
        TensorRTCvOCREngine engine{};
    public:
        TensorRTCvOCRFramework(const char* config_path);
        ~TensorRTCvOCRFramework()=default;

        int CreateEngine();
        int GetEngineInfo();

        int CreateCtx(TensorRTCvOCRCtx& ctx);
        // static int SetProcess(TensorRTCvOCRCtx& ctx,const char* name,TensorRTCvOCRProcesserBase* processer);
        static int RegisterInputLager(TensorRTCvOCRCtx& ctx,const char* name,const int64_t* shape,int64_t shape_len);
        static int RegisterOutputLager(TensorRTCvOCRCtx& ctx,const char* name,core::cuda::DataType dataType);
        static int PostImg(const char* name,TensorRTCvOCRCtx& ctx, void* img, int64_t len, core::cuda::DataType dataType);

        static int PostImg(const char* name,TensorRTCvOCRCtx& ctx, const cv::Mat& img,core::cuda::DataType dataType){
            return PostImg(name,ctx,img.data,static_cast<int64_t>(img.total()),dataType);
        };

        static int Forward(TensorRTCvOCRCtx& ctx);

        static int GetOutput(const char* name,TensorRTCvOCRCtx& ctx, core::cuda::TensorRTCudaData& data,core::cuda::DataType dataType);

        template<typename T>
        static int MatToProImg(T& processer,ProImg& pro_img,cv::Mat& mat){
            return processer.MatToProImg(mat,pro_img);
        }

        template <typename T>
        static int PreProcess(T& processer,ProImg& pro_img, PreParams params){
            return processer.PreProcess(pro_img,params);
        }

        template<typename T, typename... Args>
        static int PostProcess(T& processer, float* output, Args&&... args) {
            return processer.PostProcess(output, std::forward<Args>(args)...);
        }

        // static int PreProcess(const char* name,TensorRTCvOCRCtx& ctx,ProImg& pro_img, PreParams params){
            // auto processer=ctx.processer_map->find(name);
            // if (processer==ctx.processer_map->end()){
            //     return -1;
            // }
            // return PreProcess(processer->second,pro_img,params);
        // };

    };
}
#endif //GENERALDNNLIB_ZERO_TRT_CV_OCR_FRAMEWORK_H