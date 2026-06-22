//
// Created by HP on 2026/6/1.
//

#include "gdlz/ort/emd/onnx_rt_emd_export.h"

#include "gdlz/except_utils.h"

gdlz::ort::emd::OnnxRtEmdFramework* gdlz::ort::emd::CreateEmdFramework(const char* config_path)
{
    return new OnnxRtEmdFramework(config_path);
}

void gdlz::ort::emd::DestroyEmdFramework(OnnxRtEmdFramework* framework)
{

    delete framework;
}

void gdlz::ort::emd::GetEmdFrameworkEngineInfo(OnnxRtEmdFramework* framework)
{
    framework->GetEngineInfo();
}

int gdlz::ort::emd::SetShape(core::shape::OnnxRtShape& shape, data::OnnxRtEmdCtx& ctx)
{
    return except::try_catch([&](){
        return OnnxRtEmdFramework::SetShape(shape,ctx);
    });
}

int gdlz::ort::emd::Forward(const OnnxRtEmdFramework* framework, data::OnnxRtEmdCtx& ctx, data::OnnxRtEmdData& data)
{
    return except::try_catch([&](){
        return framework->Forward(ctx, data);
    });
}
