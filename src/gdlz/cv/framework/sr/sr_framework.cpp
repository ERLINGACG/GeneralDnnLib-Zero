//
// Created by HP on 2025/12/11.
//
#include "gdlz/cv/framework/sr/sr_framework.h"
using gdlz::cv_framework::sr::SrFramework;

SrFramework& SrFramework::ResourceDirector(resources::SrResourceDirector& resource_director)
{
        resource_director.SetConfPath(this->conf_path).Hand(this->resource);
        return *this;
}

SrFramework& SrFramework::Upsample(const param::BaseParam& param)
{
        this->resource.sr.upsample(*param.input,*param.input);
        return *this;
}
