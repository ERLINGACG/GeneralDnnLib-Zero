#include "gdlz/cv/framework/yolo_framework.h"
//
// Created by HP on 2025/12/9.
//
int main()
{

    gdlz::cv_framework::YoloFramework yolo_framework("./invoke.json");
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