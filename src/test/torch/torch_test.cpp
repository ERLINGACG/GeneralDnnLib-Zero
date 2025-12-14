//
// Created by HP on 2025/12/15.
//
#include <torch/csrc/api/include/torch/cuda.h>
#include <torch/script.h>
#include <iostream>
using namespace std;
// 临时关闭C4267警告
#pragma warning(push)  // 保存当前警告状态
#pragma warning(disable: 4267)
// 强制初始化CUDA上下文（关键：绕过自动检测）
// void force_init_cuda() {
//     // 手动加载CUDA驱动
//     if (!torch::cuda::is_available()) {
//         // 强制初始化CUDA设备
//         // 触发CUDA上下文创建
//         torch::Tensor dummy = torch::ones({1}, torch::kCUDA);
//         std::cout << "Forced CUDA init success! Dummy tensor device: " << dummy.device() << std::endl;
//     }
// }
auto test1()
{
    try
    {
        // std::cout << "LibTorch Version: " << TORCH_VERSION_MAJOR << "."
        //         << TORCH_VERSION_MINOR << "." << TORCH_VERSION_PATCH << std::endl;
        std::cout << "CUDA Available: " << torch::cuda::is_available() << std::endl;
        std::cout << "CUDA Device Count: " << torch::cuda::device_count() << std::endl;

        torch::Device device(torch::kCUDA);
        torch::jit::Module module=torch::jit::load("yolo11s.torchscript",device);
        cout<<"load model success"<<endl;
        module.eval(); // 必须设置为评估模式

        torch::Tensor input = torch::randn(
            {1, 3, 640, 640},
            torch::kFloat32).to(device) / 255.0f;

        // 3. 前向推理（包装成std::vector传入forward）
        vector<torch::jit::IValue> inputs;
        inputs.emplace_back(input);
        torch::Tensor output;
        for (int i=0; i < 10; i++)
        {
          auto start = chrono::high_resolution_clock::now();
          output=  module.forward(inputs).toTensor();
            auto end = chrono::high_resolution_clock::now();
            auto elapsed_ms = chrono::duration_cast<chrono::duration<double, milli>>(end - start).count();
            cout<<"Elapsed time: "<<elapsed_ms<<" ms\n";
        }


        // 4. 打印输出（YOLO11输出shape通常是[1, 84, 8400]，对应检测结果）
        cout << "output shape: " << output.sizes() << endl;

    }catch(std::exception& e){
        cout<<"load model fail:"<<e.what()<<endl;
    }

}
auto test2()
{
    cout<<torch::cuda::is_available()<<endl; // CUDA是否可用
}
int main()
{
    // force_init_cuda();
    test1();


    return 0;
}
