#include "../../../include/gdlz/cv/framework/yolo/yolo_framework.h"
#include "gdlz/cv/framework/sr/sr_framework.h"
//
// Created by HP on 2025/12/9.
//
int yolo_test()
{
    gdlz::cv_framework::yolo::YoloFramework yolo_framework("./invoke.json");
    gdlz::resources::YoloResourceDirector yolo_resource;
    yolo_framework.ResourceDirector(yolo_resource);

    auto img=cv::imread("img/bus.jpg");
    if(img.empty()){
        // dylog::logger().error("img is empty");
        return -1;
    }

    gdlz::cv_framework::data::YoloOutput output;
    gdlz::cv_framework::param::YoloParam yolo_param;
    yolo_framework
        .Input(img,yolo_param)
        .SetBlob(yolo_param)
        .Forward(yolo_param)
        .Process(yolo_param)
        .PostProcess(yolo_param)
        .GetOutput(output,yolo_param);

    cv::imshow("img/bus.jpg", img);
    std::cout<<*yolo_param.describe_str<<std::endl;
    std::cout<<output.json_str.get()<<std::endl;
    std::cout<<output.json_size<<std::endl;
    cv::waitKey();
    return 0;
}
int sr_test()
{
    auto img=cv::imread("img/bus.jpg");
    if(img.empty()){
        // dylog::logger().error("img is empty");
        return -1;
    }
    gdlz::cv_framework::param::BaseParam param;
    param.input=std::make_unique<cv::Mat>(img);
    gdlz::cv_framework::sr::SrFramework sr_framework("./sr.json");
    gdlz::resources::SrResourceDirector sr_resource;
    sr_framework.ResourceDirector(sr_resource).Upsample(param);
    std::cout << param.input->rows << " " << param.input->cols << std::endl;
    return 0;
}

int sr_yolo_test()
{
    gdlz::cv_framework::param::YoloParam param;
    auto img=cv::imread("img/bus.jpg");
    if(img.empty()){
        // dylog::logger().error("img is empty");
        return -1;
    }
    param.input=std::make_unique<cv::Mat>(img);

    gdlz::cv_framework::yolo::YoloFramework yolo_framework("./invoke.json");
    gdlz::resources::YoloResourceDirector yolo_resource;
    yolo_framework.ResourceDirector(yolo_resource);

    gdlz::cv_framework::sr::SrFramework sr_framework("./sr.json");
    gdlz::resources::SrResourceDirector sr_resource;
    sr_framework.ResourceDirector(sr_resource).Upsample(param);

    yolo_framework.Input(*param.input,param)
        .SetBlob(param)
        .Forward(param)
        .Process(param)
        .PostProcess(param);

    cv::imshow("img/bus.jpg", *param.input);
    std::cout<<*param.describe_str<<std::endl;
    std::cout << param.input->rows << " " << param.input->cols << std::endl;
    cv::waitKey();
    return 0;
}

auto test_make_yolo()
{
    using gdlz::cv_framework::yolo::YoloFramework;
    using gdlz::resources::YoloResourceDirector;


    YoloResourceDirector yolo_resource_director;
    YoloFramework* yolo = new YoloFramework("./invoke.json");
    yolo->ResourceDirector(yolo_resource_director);
    // 3. 返回堆指针（非悬空）
    return yolo;



}
int main()
{
    // sr_test();
    // sr_yolo_test();
    auto yolo_test=test_make_yolo();
    gdlz::cv_framework::param::YoloParam param;
    auto img=cv::imread("img/bus.jpg");
    if(img.empty()){
        return -1;
    }
    yolo_test->Input(img,param)
            .SetBlob(param)
            .Forward(param)
            .Process(param)
            .PostProcess(param);
    cv::imshow("img/bus.jpg", img);
    std::cout<<*param.describe_str<<std::endl;
    cv::waitKey();
    // delete yolo_test;
    return 0;
}