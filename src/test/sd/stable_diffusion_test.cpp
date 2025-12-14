#include <iostream>
#include <stable-diffusion.h>
// 引入 OpenCV 头文件
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

// 加载模型并返回上下文指针
sd_ctx_t* load(sd_ctx_params_t& ctx_params) {
    sd_ctx_params_init(&ctx_params); // 初始化默认值

    // 配置模型路径（根据实际模型文件路径修改）
    ctx_params.model_path = "E:/ZeroPlan/python/dnn/sd-cli/models/stable-diffusion-v1-5-Q4_0.gguf";

    // 配置计算参数
    ctx_params.n_threads = 4;
    ctx_params.wtype = SD_TYPE_F16;
    ctx_params.rng_type = CUDA_RNG;
    ctx_params.offload_params_to_cpu = false;

    sd_ctx_t* ctx = new_sd_ctx(&ctx_params);
    if (ctx == nullptr) {
        std::cerr << "Failed to create SD context." << std::endl;
        return nullptr; // 返回空指针表示失败
    }
    std::cout << "Created SD context." << std::endl;
    return ctx; // 返回成功创建的上下文
}

// 生成并保存图像（OpenCV 版本）
int sendImage(sd_ctx_t* ctx) {
    if (ctx == nullptr) {
        std::cerr << "Invalid SD context" << std::endl;
        return -1;
    }

    sd_img_gen_params_t gen_params;
    sd_img_gen_params_init(&gen_params);

    // 配置生成参数
    // 正面提示词：精准描述科比+直升机+场景+画质
    gen_params.prompt = "photorealistic 8K ultra detailed, Kobe Bryant (wearing Los Angeles Lakers 24 purple jersey), sitting in the cockpit of a modern helicopter, holding the control stick, looking forward, realistic facial expression, clear eyes, natural skin texture, helicopter interior with detailed instruments, blue sky background, soft natural lighting, cinematic composition, sharp focus, no distortion";

    // 负面提示词：过滤所有抽象/变形/低质问题
    gen_params.negative_prompt = "blurry, low resolution, abstract, deformed face, extra limbs, ugly, cartoon, anime, text, watermark, signature, distortion, messy details, overexposed, underexposed, blurry background, unrealistic proportions, weird hands, missing fingers";
    gen_params.width = 512;
    gen_params.height = 512;
    gen_params.seed = 12345;
    gen_params.batch_count = 1;

    // 配置采样参数
    gen_params.sample_params.sample_steps = 30;
    gen_params.sample_params.sample_method = EULER_A_SAMPLE_METHOD;
    gen_params.sample_params.scheduler = KARRAS_SCHEDULER;
    gen_params.sample_params.guidance.txt_cfg = 7.5f;

    // 执行生成
    sd_image_t* image = generate_image(ctx, &gen_params);
    if (image == nullptr) {
        std::cerr << "Failed to generate image" << std::endl;
        return -1;
    }

    // ========== 核心：用 OpenCV 处理并保存图像 ==========
    try {
        // 1. 将 sd_image_t 的数据转换为 OpenCV Mat
        // 注意：sd_image_t 的数据格式通常是 RGB 通道（uint8_t），而 OpenCV 默认是 BGR
        cv::Mat cv_image;
        if (image->channel == 3) {
            // RGB 转 BGR（OpenCV 显示/保存需要 BGR 格式）
            cv_image = cv::Mat(image->height, image->width, CV_8UC3, image->data);
            cv::cvtColor(cv_image, cv_image, cv::COLOR_RGB2BGR);
        } else if (image->channel == 4) {
            // RGBA 转 BGRA（带透明通道）
            cv_image = cv::Mat(image->height, image->width, CV_8UC4, image->data);
            cv::cvtColor(cv_image, cv_image, cv::COLOR_RGBA2BGRA);
        } else if (image->channel == 1) {
            // 灰度图
            cv_image = cv::Mat(image->height, image->width, CV_8UC1, image->data);
        }

        // 2. 保存图像（支持 PNG/JPG/JPEG 等格式）
        bool save_success = cv::imwrite("output_opencv2.png", cv_image);
        if (save_success) {
            std::cout << "✅ image saved as output_opencv_3.png" << std::endl;

            // 可选：显示生成的图像（需要有图形界面）
            // cv::imshow("Generated Image", cv_image);
            // cv::waitKey(0); // 等待按键关闭窗口
            // cv::destroyAllWindows();
        } else {
            std::cerr << "❌ OpenCV 保存图像失败" << std::endl;
        }
    } catch (const cv::Exception& e) {
        std::cerr << "❌ OpenCV 处理图像出错：" << e.what() << std::endl;
    }

    // 释放图像资源
    free(image->data);
    free(image);
    return 0;
}

void benchmark_generation(sd_ctx_t* ctx) {
    sd_img_gen_params_t gen_params;
    sd_img_gen_params_init(&gen_params);

    // 配置参数
    gen_params.prompt = "test";
    gen_params.width = 512;
    gen_params.height = 512;
    gen_params.seed = 123;
    gen_params.sample_params.sample_steps = 20;

    // 测试3次取平均
    double total_time = 0;
    int runs = 3;

    for(int i = 0; i < runs; i++) {
        auto start = std::chrono::high_resolution_clock::now();

        sd_image_t* image = generate_image(ctx, &gen_params);

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        total_time += elapsed;

        std::cout << "运行 " << i+1 << ": " << elapsed << "秒" << std::endl;

        if(image) {
            free(image->data);
            free(image);
        }
    }

    std::cout << "平均时间: " << total_time/runs << "秒" << std::endl;
}
int main() {
    sd_ctx_params_t ctx_params;
    sd_ctx_t* ctx = load(ctx_params); // 获取上下文指针

    if (ctx != nullptr) {
        auto start = std::chrono::high_resolution_clock::now();
        sendImage(ctx); // 调用生成函数
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "send image: " << elapsed << "s" << std::endl;
        // 基准测试
        // benchmark_generation(ctx);

        free_sd_ctx(ctx); // 释放上下文资源
    }

    return 0;
}