//
// Created by HP on 2025/12/9.
//
// #include "../../../../include/gdlz/cv/framework/yolo/yolo_resources.h"
#include "gdlz/cv/framework/yolo/yolo_resources.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <dylog/logger.h>
using gdlz::resources::CV_YoloResourceDirector;
using nlohmann::json;
CV_YoloResourceDirector& CV_YoloResourceDirector::Hand(YoloResource& resource)
{
    using std::unique_ptr;
    using std::ifstream;

    auto logger=dylog::DynamicLogger().setInvokeName("CV_YoloResourceDirector");

    auto set_env=[&](json j,json& env)
    {
        env=j[j["env"].get<std::string>()];
        logger.debug("use env: {}",j["env"].get<std::string>().c_str());
    };

    auto set_mode_path=[&](json& env)
    {
        resource.model_path=env["model_path"].get<std::string>();
        logger.debug("model_path: {}",resource.model_path.c_str());
    };

    auto set_model_conf_thresh=[&](json& env)
    {
        resource.conf_thresh=env["conf_thresh"].get<float>();
    };

    auto set_model_nms_thresh=[&](json& env)
    {
        resource.nms_thresh=env["nms_thresh"].get<float>();
    };

    auto set_model_input_shape=[&](json& env)
    {
        resource.input_height=env["input_height"].get<int>();
        resource.input_width=env["input_width"].get<int>();
    };

    auto set_model_is_fusion=[&](json& env)
    {
        resource.is_fusion=env["is_fusion"].get<bool>();
    };

    auto set_model_class_names=[&](json& env)
    {
        resource.class_names=std::make_unique<std::vector<std::string>>(env["class_names"].get<std::vector<std::string>>());
        logger.debug("class_names size: {}",resource.class_names->size());
    };

    auto set_model_enable_backend=[&](json& env)
    {
        resource.enable_backend=env["enable_backend"].get<int>();
    };

    auto load_model=[&](){
        resource.Net=std::make_unique<cv::dnn::Net>(cv::dnn::readNet(resource.model_path));

        if (resource.Net->empty()) {
            logger.error("load model failed: {}",resource.model_path.c_str());
        }else {
            logger.info("load model success: {}",resource.model_path.c_str());
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
                logger.info("enable backend cuda_fp16");
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
            logger.info("enable fusion: {}",resource.is_fusion);
        }
    };


    try {
        const unique_ptr<ifstream> jsonfile(new ifstream(this->conf_path));
        json config;
        * jsonfile >> config;
        json env;

        set_env(config,env);
        set_mode_path(env);
        set_model_input_shape(env);
        set_model_is_fusion(env);
        set_model_enable_backend(env);
        set_model_conf_thresh(env);
        set_model_nms_thresh(env);
        set_model_class_names(env);
        load_model();

    }catch (std::exception& e) {
        logger.error("Hand: exception: {}", e.what());
    }
    catch(...) {
       logger.error("Hand: unknown exception");
    }
    return *this;
}

CV_YoloResourceDirector& CV_YoloResourceDirector::SetConfPath(const std::string& conf_path)
{
    this->conf_path = conf_path;
    return *this;
}
