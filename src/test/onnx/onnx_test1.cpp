#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include "opencv2/opencv.hpp"
using namespace std;

struct CDetectBox
{
    int id;
    float score;
    float x1;
    float y1;
    float x2;
    float y2;
};

float compute_iou(const CDetectBox& box1, const CDetectBox& box2)
{
    float xmin = std::max(box1.x1, box2.x1);
    float ymin = std::max(box1.y1, box2.y1);
    float xmax = std::min(box1.x2, box2.x2);
    float ymax = std::min(box1.y2, box2.y2);

    float intersection_area = std::fmax(0.0, xmax - xmin) * std::fmax(0.0, ymax - ymin);
    float boxArea1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    float boxArea2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
    float union_area = boxArea1 + boxArea2 - intersection_area + 1e-10;

    return intersection_area / union_area;
}

int main()
{
    int imgH = 640;
    int imgW = 640;

    cv::Mat image = cv::imread("img/bus.jpg", 1);
    cv::resize(image, image, cv::Size(imgW, imgH));

    vector<float> imgData;
    imgData.resize(3 * imgW * imgH);
    for (int k = 0; k < 3; k++)
    {
        for (int i = 0; i < imgH; i++)
        {
            for (int j = 0; j < imgW; j++)
            {
                imgData[k * imgW * imgH + i * imgW + j] = image.at<cv::Vec3b>(i, j)[3 - k] / 255.0;
            }
        }
    }

    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "yolo11");
    OrtCUDAProviderOptions cudaOptions;
    cudaOptions.device_id = 0;
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    // sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    static Ort::Session yolo_session(env, L"./model/yolo11s.onnx", sessionOptions);

    vector<int64_t> inputShape{ 1, 3, imgH, imgW };

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, imgData.data(), imgData.size(), inputShape.data(), inputShape.size());

    const char* inputNames[] = { "images" };
    const char* outputNames[] = { "output0" };
    Ort::RunOptions runOptions;
    vector<Ort::Value> out;
    for (int iz=0;iz<10;iz++)
    {
        auto start = chrono::high_resolution_clock::now();
       out  = yolo_session.Run(runOptions, inputNames, &inputTensor, 1, outputNames, 1);
       auto end = chrono::high_resolution_clock::now();
        auto elapsed_ms = chrono::duration_cast<chrono::duration<double, milli>>(end - start).count();
      cout<<"Elapsed time: "<<elapsed_ms<<" ms\n";

    }

    std::vector<int64_t> shape = out[0].GetTensorTypeAndShapeInfo().GetShape();
    float* data = (float*)out[0].GetTensorMutableData<void>();  //data shape {1,84,8400}

    /////////////////////////////postprocess////////////////////////////////////////////
    std::vector<CDetectBox> allBoxes, nmsBoxes;
    allBoxes.reserve(200);
    for (int w = 0; w < 8400; w++)
    {
        float maxScore = 0;
        int id = -1;
        for (int h = 4; h < 84; h++)
        {
            if (maxScore < data[h * 8400 + w])
            {
                maxScore = data[h * 8400 + w];
                id = h - 4;
            }
        }

        if (maxScore > 0.3)
        {
            CDetectBox box;
            box.id = id;
            box.score = maxScore;
            box.x1 = data[0 * 8400 + w] - data[2 * 8400 + w] / 2;
            box.y1 = data[1 * 8400 + w] - data[3 * 8400 + w] / 2;
            box.x2 = data[0 * 8400 + w] + data[2 * 8400 + w] / 2;
            box.y2 = data[1 * 8400 + w] + data[3 * 8400 + w] / 2;
            allBoxes.emplace_back(box);
        }
    }

    std::sort(allBoxes.begin(), allBoxes.end(), [](CDetectBox& a, CDetectBox& b) { return a.score > b.score; });

    while (!allBoxes.empty())        //nms
    {
        nmsBoxes.emplace_back(allBoxes.front());
        auto it = allBoxes.begin();
        allBoxes.erase(it);

        for (auto it = allBoxes.begin(); it != allBoxes.end();)
        {
            float iou = compute_iou(nmsBoxes.back(), *it);
            if (iou > 0.5)
            {
                it = allBoxes.erase(it);
            }
            else
            {
                it++;
            }
        }
    }
    /////////////////////////////////////////////////////////////////////////

    for (int i = 0; i < nmsBoxes.size(); i++)
    {
        int x = nmsBoxes[i].x1;
        int y = nmsBoxes[i].y1;
        int w = nmsBoxes[i].x2 - nmsBoxes[i].x1;
        int h = nmsBoxes[i].y2 - nmsBoxes[i].y1;

        printf("%d %f %f %f %f %f\n", nmsBoxes[i].id, nmsBoxes[i].score, nmsBoxes[i].x1, nmsBoxes[i].y1, nmsBoxes[i].x2, nmsBoxes[i].y2);
        cv::Rect rect(x, y, w, h);
        cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 1, 8, 0);
    }
    cv::imshow("img", image);
    cv::waitKey(0);
    return 0;
}