//
// Created by HP on 2026/5/14.
//
#include <opencv2/dnn_superres.hpp>
#include <opencv2/opencv.hpp>

auto load(const std::string& model_path){
    auto sr = cv::dnn_superres::DnnSuperResImpl();
    try {

        sr.readModel(model_path);
        sr.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        sr.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        sr.setModel("fsrcnn", 2);
        return sr;
    }catch (std::exception& e) {
        std::cerr << "load model failed: " << e.what() << std::endl;
        return sr;
    }
}

int main(const int argc, char* argv[]){

    std::string model_path;
    std::string video_path;

    for (int i = 1; i < argc; ++i) {  // 跳过第0个（程序名）
        std::string arg = argv[i];
        if (arg.find("--model=") == 0) {
            model_path = arg.substr(8);  // 截取 "--model=" 之后的部分
        } else if (arg.find("--video=") == 0) {
            video_path = arg.substr(8);  // 截取 "--video=" 之后的部分
        }
    }

    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Video: " << video_path << std::endl;

    auto sr=load(model_path);
    cv::VideoCapture cap(video_path);
    cv::Mat frame;
    if (!cap.isOpened()){
        std::cerr << "open video failed: " << video_path << std::endl;
        return -1;
    }

    while (cap.isOpened()){
        cap >> frame;
        if (frame.empty()){
            break;
        }
        sr.upsample(frame,frame);
        cv::imshow("frame",frame);
        std::cout << "frame: " << frame.cols << "x" << frame.rows << std::endl;
        cv::waitKey(1);

    }

    return 0;
}
