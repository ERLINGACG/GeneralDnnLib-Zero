//
// Created by HP on 2026/1/13.
//
#include "gdlz/cv/framework/face/base/feature_dnn_resources_director.h"

#include <fstream>
#include <dylog/logger.h>
#include <nlohmann/json.hpp>
using nlohmann::json;
gdlz::resources::CV_FeatureDnnResourceDirector& gdlz::resources::CV_FeatureDnnResourceDirector::SetConfPath(const std::string& conf_path)
{
    this->conf_path=conf_path;
    return *this;
}

gdlz::resources::CV_FeatureDnnResourceDirector& gdlz::resources::CV_FeatureDnnResourceDirector::Hand(FeatureDnnResource& resource)
{


    auto logger =dylog::DynamicLogger().setInvokeName("CV_FeatureDnnResourceDirector");

    auto set_env=[&](const json& j,json& env)
    {

        env=j[j["env"].get<std::string>()];
        logger.debug("use env: {}",j["env"].get<std::string>().c_str());

    };
    auto set_mode_path=[&](const json& env)
    {
        resource.model_path=env["model_path"].get<std::string>();
        logger.debug("use model_path: {}",resource.model_path.c_str());
    };

    auto set_is_fusion=[&](const json& env)
    {
        resource.is_fusion=env["is_fusion"].get<bool>();
        logger.debug("use is_fusion: {}",resource.is_fusion);
    };

    auto set_input_size=[&](const json& env)
    {
        resource.input_height=env["input_height"].get<int>();
        resource.input_width=env["input_width"].get<int>();
        logger.debug("use input_size: {}x{}",resource.input_height,resource.input_width);
    };

    auto set_backend=[&](const json& env)
    {
        resource.enable_backend=env["enable_backend"].get<int>();
        logger.debug("use enable_backend: {}",resource.enable_backend);
    };

    auto load_mode=[&]()
    {

        resource.Net=std::make_unique<cv::dnn::Net>(cv::dnn::readNet(resource.model_path));
        switch (resource.enable_backend)
        {
            case 0:
                resource.Net->setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                resource.Net->setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
                logger.info("enable backend cpu");
                break;
            case 1:
                resource.Net->setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                resource.Net->setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
                logger.info("enable backend cuda");
                break;
            case 2:
                resource.Net->setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                resource.Net->setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
                logger.info("enable backend cuda fp16");
                break;
            case 3:
                resource.Net->setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                resource.Net->setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
                logger.info("enable backend opencl");
                break;
            case 4:
                resource.Net->setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
                resource.Net->setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
                logger.info("enable backend openvino");
                break;

            default:
                resource.Net->setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                resource.Net->setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
                logger.warn("enable_backend: {} is not support,use the default: CPU",resource.enable_backend);
                break;
        }
        resource.Net->enableFusion(resource.is_fusion);
        if (resource.Net->empty()) {
            logger.error("load model {} failed",resource.model_path);
        }else
        {
            logger.info("load model success to : {}",resource.model_path);
        }
    };

    try
    {
        using std::unique_ptr;
        using std::make_unique;
        using std::ifstream;
        const auto jsonfile=make_unique<std::ifstream>(this->conf_path);

        json j;
        *jsonfile >> j;
        json env;

        set_env(j,env);
        set_mode_path(env);
        set_is_fusion(env);
        set_input_size(env);
        set_backend(env);
        load_mode();

        // logger.info("load model {} success",resource.model_path);
    }catch (std::exception& e) {
        logger.error("load model {} failed, error: {}",resource.model_path,e.what());
    }
    return *this;
}
