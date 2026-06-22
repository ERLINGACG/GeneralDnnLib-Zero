//
// Created by HP on 2026/5/3.
//
#include "gdlz/ort/core/onnx_rt_core_export.h"

#include <iostream>

gdlz::ort::core::OnnxRtFramework* gdlz::ort::core::CreateOnnxRtFramework(const char* config_path)
{
    return (new OnnxRtFramework())->ResourceDirector(config_path);
}

int gdlz::ort::core::CreateEngine(const OnnxRtFramework* framework, data::OnnxRTEngine& ctx)
{
    return framework->CreateEngine(ctx);
}

void gdlz::ort::core::GetEngineInfo(const OnnxRtFramework* framework, data::OnnxRTEngine& ctx)
{
    framework->GetEngineInfo(ctx);
}



void  gdlz::ort::core::Run(const OnnxRtFramework* framework, data::OnnxRTEngine& ctx)
{
    framework->run(ctx);
}

int gdlz::ort::core::RunExam(const OnnxRtFramework* framework, void* token, int len, data::OnnxRTEngine& ctx)
{
    //
    auto ikv=data::OnnxExamKV();
    framework->run_batch_init(token, len, ctx, ikv);
    for (int i=0; i<120; i++) {
        framework->run_token_one_step(ctx, ikv);
    }

    return 0;
}
