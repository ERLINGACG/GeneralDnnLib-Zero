//
// Created by HP on 2026/1/6.
//

#include "gdlz/cv/framework/face/yunet/yunet_resources.h"
#include <memory>
#include <nlohmann/json.hpp>
#include <fstream>
#include <dylog/logger.h>
using gdlz::resources::CV_YuNetResourceDirector;
using nlohmann::json;
using std::unique_ptr;
using std::ifstream;
CV_YuNetResourceDirector& CV_YuNetResourceDirector::Hand(YuNetResource& resource)
{

    auto logger=dylog::DynamicLogger().setInvokeName("CV_YuNetResourceDirector");
    auto set_env=[&](json j,json& env)
    {
        env=j[j["env"].get<std::string>()];
        logger.debug("use env: {}",j["env"].get<std::string>().c_str());
    };

    auto set_model_path=[&](json env)
    {
        resource.model_path=env["model_path"].get<std::string>();
        logger.debug("use model_path: {}",resource.model_path.c_str());
    };

    auto set_backend=[&](json env)
    {
        resource.backend=env["enable_backend"].get<int>();
        logger.debug("use backend: {}",resource.backend);
    };
    auto set_max_face=[&](json env)
    {
        resource.max_Face=env["max_face"].get<int>();
        logger.debug("use max_Face: {}",resource.max_Face);
    };


    auto set_thresh=[&](json env)
    {
        resource.conf_thresh=env["conf_thresh"].get<float>();
        logger.debug("use conf_thresh: {}",resource.conf_thresh);

        resource.nms_thresh=env["nms_thresh"].get<float>();
        logger.debug("use nms_thresh: {}",resource.nms_thresh);
    };


    auto set_input_size=[&](json env)
    {
        resource.input_height=env["input_height"].get<int>();
        resource.input_width=env["input_width"].get<int>();
        logger.debug("use input_size: {}x{}",resource.input_height,resource.input_width);
    };


    auto load_model=[&]()
    {
        // resource.detector=cv::FaceDetectorYN::create(
        //     )
        switch (resource.backend) {
            case 0:
                resource.detector=cv::FaceDetectorYN::create(
                    resource.model_path,
                    "",
                    cv::Size(resource.input_width,resource.input_height),
                    resource.conf_thresh,
                    resource.nms_thresh,
                    resource.max_Face,
                    cv::dnn::Backend::DNN_BACKEND_OPENCV,
                    cv::dnn::Target::DNN_TARGET_CPU
                );
                logger.debug("user the CPU backend");
                break;
            case 1:
                   resource.detector=cv::FaceDetectorYN::create(
                       resource.model_path,
                       "",
                       cv::Size(resource.input_width,resource.input_height),
                       resource.conf_thresh,
                       resource.nms_thresh,
                       resource.max_Face,
                       cv::dnn::Backend::DNN_BACKEND_CUDA,
                       cv::dnn::Target::DNN_TARGET_CUDA
                   );
                logger.debug("user the CUDA backend");
                break;
            case 2:
                   resource.detector=cv::FaceDetectorYN::create(
                       resource.model_path,
                       "",
                       cv::Size(resource.input_width,resource.input_height),
                       resource.conf_thresh,
                       resource.nms_thresh,
                       resource.max_Face,
                       cv::dnn::Backend::DNN_BACKEND_CUDA,
                       cv::dnn::Target::DNN_TARGET_CUDA_FP16
                   );
                logger.debug("user the CUDA_FP16 backend");
                break;

            case 3:
                  resource.detector=cv::FaceDetectorYN::create(
                      resource.model_path,
                       "",
                       cv::Size(resource.input_width,resource.input_height),
                       resource.conf_thresh,
                       resource.nms_thresh,
                       resource.max_Face,
                       cv::dnn::Backend::DNN_BACKEND_OPENCV,
                       cv::dnn::Target::DNN_TARGET_OPENCL
                   );
                logger.debug("user the OpenCL backend");
                break;
            case 4:
                   resource.detector=cv::FaceDetectorYN::create(
                       resource.model_path,
                       "",
                       cv::Size(resource.input_width,resource.input_height),
                       resource.conf_thresh,
                       resource.nms_thresh,
                       resource.max_Face,
                       cv::dnn::Backend::DNN_BACKEND_INFERENCE_ENGINE,
                       cv::dnn::Target::DNN_TARGET_CPU
                   );
                logger.debug("user the INFERENCE_ENGINE backend");
                break;
            default:
                resource.detector=cv::FaceDetectorYN::create(
                    resource.model_path,
                    "",
                    cv::Size(resource.input_width,resource.input_height),
                    resource.conf_thresh,
                    resource.nms_thresh,
                    resource.max_Face,
                    cv::dnn::Backend::DNN_BACKEND_OPENCV,
                    cv::dnn::Target::DNN_TARGET_CPU
                );
                logger.warn("backend not support: {}, use default backend",resource.backend);
                break;
        }
        if(resource.detector.empty()) {
            logger.error("load model failed: {}",resource.model_path.c_str());
        }else{
            logger.info("load model success: {}",resource.model_path.c_str());
        }
    };

    try {

        const unique_ptr<ifstream> jsonfile = std::make_unique<ifstream>(this->conf_path);
        json config;
        * jsonfile >> config;
        json env;
        set_env(config,env);
        set_backend(env);
        set_thresh(env);
        set_input_size(env);
        set_model_path(env);
        set_max_face(env);
        load_model();
    }catch (std::exception& e) {
        logger.error("load config failed: {}",e.what());
    }
    return *this;
}

CV_YuNetResourceDirector& CV_YuNetResourceDirector::SetConfigPath(const std::string& config_path)
{

    this->conf_path=config_path;
    return *this;
}
