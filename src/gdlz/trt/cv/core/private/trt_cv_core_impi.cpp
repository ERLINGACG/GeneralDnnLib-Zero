//
// Created by HP on 2026/10/19.
//
#include "gdlz/trt/cv/core/private/trt_cv_core_impi.h"
#include "gdlz/trt/cv/core/trt_cv_core_data.h"
using gdlz::trt::icv::core::TensorRTCvImpi;
using gdlz::trt::icv::core::TensorRTCvCtx;
using gdlz::trt::icv::core::TensorRTCvEngine;

TensorRTCvEngine::TensorRTCvEngine(): engine_impi(std::make_unique<TensorRTCvEngineImpi>()) {}
TensorRTCvEngine::~TensorRTCvEngine() = default;
TensorRTCvEngine::TensorRTCvEngine(TensorRTCvEngine&&) noexcept = default;
TensorRTCvEngine& TensorRTCvEngine::operator=(TensorRTCvEngine&&) noexcept = default;

TensorRTCvCtx::TensorRTCvCtx(): impi_ctx(std::make_unique<TensorRTCvImpi>()) {}

TensorRTCvCtx::~TensorRTCvCtx() = default;                  // ★ 只有这里才看得到 Impl 完整定义
TensorRTCvCtx::TensorRTCvCtx(TensorRTCvCtx&&) noexcept = default;
TensorRTCvCtx& TensorRTCvCtx::operator=(TensorRTCvCtx&&) noexcept = default;



