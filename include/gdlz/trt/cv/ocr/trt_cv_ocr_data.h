//
// Created by HP on 2026/7/2.
//

#ifndef GENERALDNNLIB_ZERO_TRT_CV_OCR_DATA_H
#define GENERALDNNLIB_ZERO_TRT_CV_OCR_DATA_H
#include <memory>

#include "trt_ocr_process.h"
#include "gdlz/trt/core/trt_core_data.h"

namespace gdlz::trt::icv::ocr {




    struct TensorRTCvOCREngine{
        gdlz::trt::core::TensorRTCoreEngine engine{};

        auto& getCoreEngine(){
            return engine;
        }

    };



    struct TensorRTCvOCRCtx {
        // int64_t seq_len;
        std::unique_ptr<gdlz::trt::core::TensorRTCoreCtx>      core_ctx;
        // std::unique_ptr<std::map<std::string, TensorRTCvOCRProcesserBase*>> processer_map;

        [[nodiscard]] auto& getCoreCtx() const{
            return *core_ctx;
        }
    };
}
#endif //GENERALDNNLIB_ZERO_TRT_CV_OCR_DATA_H