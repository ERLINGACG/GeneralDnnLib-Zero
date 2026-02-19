//
// Created by HP on 2026/2/13.
//
#include <fstream>
#include <iostream>
#include <NvInfer.h>

#include <vector>
#include <NvInferRuntime.h>
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // 只打印 WARNING 及以上级别的日志
        if (severity <= Severity::kWARNING) {
            std::cout << "TRT_LOG: " << msg << std::endl;
        }
    }
};

// void doInference(IExecutionContext& context, float* input, float* output, int batchSize, int inputSize, int outputSize)
// {
//     const ICudaEngine& engine = context.getEngine();
//     assert(engine.getNbBindings() == 2);
//
//     // Allocate GPU buffers
//     void* buffers[2];
//     const int inputIndex = engine.getBindingIndex("images");
//     const int outputIndex = engine.getBindingIndex("output");
//     buffers[inputIndex] = safeCudaMalloc(inputSize * batchSize * sizeof(float));
//     buffers[outputIndex] = safeCudaMalloc(outputSize * batchSize * sizeof(float));
//
//     // Create stream
//     cudaStream_t stream;
//     cudaStreamCreate(&stream);
//
//     // Copy input data to GPU
//     cudaMemcpyAsync(buffers[inputIndex], input, inputSize * batchSize * sizeof(float), cudaMemcpyHostToDevice, stream);
//
//     // Execute inference
//     context.enqueue(batchSize, buffers, stream, nullptr);
//
//     // Copy output data to CPU
//     cudaMemcpyAsync(output, buffers[outputIndex], outputSize * batchSize * sizeof(float), cudaMemcpyDeviceToHost, stream);
//     cudaStreamSynchronize(stream);
//
//     // Release stream and buffers
//     cudaStreamDestroy(stream);
//     cudaFree(buffers[inputIndex]);
//     cudaFree(buffers[outputIndex]);
// }
int main()
{

    Logger logger;
    const std::string engine_file_path = "model.trt"; // 你的引擎文件路径
    std::ifstream file(engine_file_path, std::ios::binary | std::ios::ate);
    if (!file.good()) {
        std::cerr << "read engine file failed: " << engine_file_path << std::endl;
        return -1;
    }
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(file_size);
    file.read(engine_data.data(), static_cast<long>(file_size));
    file.close();

    std::cout << "engine file size: " << file_size << std::endl;
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    if (!runtime) {
        std::cerr << "create IRuntime failed" << std::endl;
        return -1;
    }
    std::cout << "create IRuntime success" << std::endl;

    auto engine = runtime->deserializeCudaEngine(engine_data.data(), file_size);
    if (!engine) {
        std::cerr << "blob to file failed" << std::endl;
        delete runtime;
        return -1;
    }
    std::cout << "blob to file success" << std::endl;

    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    if (context==nullptr)
    {
        std::cerr << "create ExecutionContext failed" << std::endl;
        return -1;
    }
    std::cout << "create ExecutionContext success" << std::endl;

    // std::vector<float> input(3 * 640 * 640);
    // -------------------------- 2. 定义输入参数（匹配1×3×640×640） --------------------------
    const int batch_size = 1;
    const int channel = 3;
    const int height = 640;
    const int width = 640;
    const int input_elem_num = batch_size * channel * height * width; // 1×3×640×640=1,228,800
    const int input_byte_size = input_elem_num * sizeof(float);

    // -------------------------- 3. 分配CPU/GPU内存（空输入用全0填充） --------------------------
    // CPU主机内存：创建空输入（全0）
    std::vector<float> host_input(input_elem_num, 0.0f); // 空输入：所有元素初始化为0

    // GPU设备内存：TensorRT推理必须用设备内存
    void* device_input = nullptr;

    cudaMalloc(&device_input, input_byte_size);
    nvinfer1::Dims4 input_dims = {batch_size, channel, height, width};
    context->setInputShape("images", input_dims);
    context->setTensorAddress("images", device_input);

    std::cout << "create input success" << std::endl;

    nvinfer1::Dims output_dims = context->getTensorShape("output0");
    std::cout << "output0 dims: " << std::endl;
    for (int i = 0; i < output_dims.nbDims; ++i) {
        std::cout << output_dims.d[i] << " ";
    }
    std::cout << std::endl; // 应输出：1 84 8400

    // 3. 计算输出总元素数和字节数
    int output_elem_num = 1;
    for (int i = 0; i < output_dims.nbDims; ++i) {
        output_elem_num *= output_dims.d[i];
    }
    int output_byte_size = output_elem_num * sizeof(float); // 705600 * 4 = 2822400 字节

    std::vector<float> host_output(output_elem_num, 0.0f); // CPU内存（存最终结果）
    void* device_output = nullptr;
    cudaMalloc(&device_output, output_byte_size); // GPU内存（推理时存输出）

    // 5. 绑定输出GPU内存地址到上下文
    context->setTensorAddress("output0", device_output);
    std::cout << "create output success" << std::endl;

    // -------------------------- 执行推理 + 读取输出 --------------------------
    // 1. CPU→GPU拷贝输入数据（空输入）
    cudaMemcpy(device_input, host_input.data(), input_elem_num * sizeof(float), cudaMemcpyHostToDevice);
    std::cout << "copy input success" << std::endl;


    // 1. 创建CUDA流（异步推理必备，0为默认流）
    cudaStream_t cuda_stream;
    auto cuda_status = cudaStreamCreate(&cuda_stream);
    if (cuda_status != cudaSuccess) {
        std::cerr << "create CUDA stream is fail：" << cudaGetErrorString(cuda_status) << std::endl;
        cudaFree(device_input);
        cudaFree(device_output);
        return -1;
    }

    // 2. CPU → GPU 拷贝输入数据（异步，关联到创建的流）
    cuda_status = cudaMemcpyAsync(device_input, host_input.data(), input_byte_size,
                                 cudaMemcpyHostToDevice, cuda_stream);
    if (cuda_status != cudaSuccess) {
        std::cerr << "copy input to CUDA stream is fail：" << cudaGetErrorString(cuda_status) << std::endl;
        cudaStreamDestroy(cuda_stream);
        cudaFree(device_input);
        cudaFree(device_output);
        return -1;
    }

    // 3. 调用官方enqueueV3接口执行推理（异步）
    bool infer_success = context->enqueueV3(cuda_stream);
    if (!infer_success) {
        std::cerr << "enqueueV3 inference is fail" << std::endl;
        cudaStreamDestroy(cuda_stream);
        cudaFree(device_input);
        cudaFree(device_output);
        return -1;
    }

    std::cout << "infer success" << std::endl;

    // 4. GPU → CPU 拷贝输出数据（异步，需等待推理完成）
    cuda_status = cudaMemcpyAsync(host_output.data(), device_output, output_byte_size,
                                 cudaMemcpyDeviceToHost, cuda_stream);
    if (cuda_status != cudaSuccess) {
        std::cerr << "copy output to host is fail：" << cudaGetErrorString(cuda_status) << std::endl;
        cudaStreamDestroy(cuda_stream);
        cudaFree(device_input);
        cudaFree(device_output);
        return -1;
    }

    // 5. 等待CUDA流中所有操作完成（确保数据拷贝完毕）
    cuda_status = cudaStreamSynchronize(cuda_stream);
    if (cuda_status != cudaSuccess) {
        std::cerr << "synchronize CUDA stream is fail：" << cudaGetErrorString(cuda_status) << std::endl;
        cudaStreamDestroy(cuda_stream);
        cudaFree(device_input);
        cudaFree(device_output);
        return -1;
    }
    std::cout << "copy output to host success" << std::endl;
    std::cout << "output0 elem num: " << output_elem_num << std::endl;
    std::cout << "output0前20个值：";
    for (int i = 0; i < 20; ++i) {
        std::cout << host_output[i] << " ";
    }
    std::cout << std::endl;


    // context->setTensorAddress("images", );

    return 0;
}