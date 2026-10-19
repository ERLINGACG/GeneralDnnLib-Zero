//
// Created by HP on 2026/7/8.
//

#ifndef GENERALDNNLIB_ZERO_TRT_CV_CORE_DATA_H
#define GENERALDNNLIB_ZERO_TRT_CV_CORE_DATA_H
// #include "gdlz/trt/core/trt_core_data.h"
#include <memory>
#include "gdlz/export.h"
namespace gdlz::trt::icv::core{



    struct TensorRTCvEngineImpi;

    struct GDLZ_CPP_API TensorRTCvEngine{

        std::unique_ptr<TensorRTCvEngineImpi> engine_impi;

        TensorRTCvEngine();
        ~TensorRTCvEngine();
        TensorRTCvEngine(TensorRTCvEngine&&) noexcept;
        TensorRTCvEngine& operator=(TensorRTCvEngine&&) noexcept;
        TensorRTCvEngine(const TensorRTCvEngine&)            = delete;
        TensorRTCvEngine& operator=(const TensorRTCvEngine&) = delete;

        auto& impi(){
            return engine_impi;
        }
    };


    struct TensorRTCvImpi;

    struct GDLZ_CPP_API TensorRTCvCtx{

        std::unique_ptr<TensorRTCvImpi> impi_ctx;

        TensorRTCvCtx();
        ~TensorRTCvCtx();
        TensorRTCvCtx(TensorRTCvCtx&&) noexcept;
        TensorRTCvCtx& operator=(TensorRTCvCtx&&) noexcept;
        TensorRTCvCtx           (const TensorRTCvCtx&)            = delete;
        TensorRTCvCtx& operator=(const TensorRTCvCtx&) = delete;

        auto& impi(){
            return impi_ctx;
        }

    };

}
#endif //GENERALDNNLIB_ZERO_TRT_CV_CORE_DATA_H