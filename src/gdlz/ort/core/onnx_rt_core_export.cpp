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
    OnnxRtFramework::GetEngineInfo(ctx);
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

    // int64_t token_0[1] = {first_token_ret};
    // for (int i=1; i<10; i++){
    //   int res = framework->run_token(token_0, 1, ctx);
    //   token_0[0] = res;
    // }
    return 0;
}
