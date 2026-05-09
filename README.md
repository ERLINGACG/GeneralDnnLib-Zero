# GeneralDnnLib_Zero

`GeneralDnnLib_Zero` 是一个面向 **C++ 深度学习推理与能力封装** 的实验型项目，目标是把常见的视觉、LLM 与 TensorRT 能力整理成统一的框架接口，方便在本地工程中快速接入。

当前仓库以 **Windows + CMake + MSVC** 为主，已经可以看到较完整的模块骨架与若干测试入口，适合作为个人能力库、推理框架封装样例或后续业务项目的基础设施。

## 项目定位

- 统一封装多类推理能力：CV、LLM、TensorRT
- 提供相对一致的资源加载、输入、推理、后处理接口
- 面向本地依赖库集成，而不是纯头文件开箱即用项目
- 当前更偏“工程实践仓库”而不是完全产品化 SDK

## 当前能力

### 1. 计算机视觉

- **YOLO 检测框架**
  - 头文件：`include/gdlz/cv/framework/yolo/yolo_framework.h`
  - 支持资源加载、输入、前向推理、后处理、输出提取
- **人脸检测 / 识别**
  - `YuNet`：人脸检测
  - `ArcFace`：人脸特征提取 / 识别
  - 相关工具：`face_utils`、`record`
- **图像超分模块**
  - 路径：`include/gdlz/cv/framework/sr`

### 2. LLM / GGUF

- 基于 `llama.cpp` 的 GGUF 推理封装
- 提供批处理、流式输出、上下文初始化等接口
- 支持 Embedding 相关流程
- 头文件：`include/gdlz/llm/framework/llm_gguf_framework.h`

### 3. TensorRT / RAG Embedding

- 提供 TensorRT Embedding 引擎封装
- 包含上下文创建、输入投递、维度设置、前向执行、向量提取
- 头文件：`include/gdlz/trt/rag/trt_emb_framework.h`

### 4. 预留与实验模块

- ONNX Runtime 测试入口
- LibTorch 测试入口
- Stable Diffusion 测试入口
- TensorFlow 测试入口
- 插件目录：`src/gdlz/plugin`

## 目录结构

```text
GeneralDnnLib-Zero/
├─ include/
│  └─ gdlz/
│     ├─ cv/                  # CV 框架头文件
│     ├─ llm/                 # LLM 框架头文件
│     ├─ res/                 # 资源定义
│     └─ trt/                 # TensorRT 相关头文件
├─ src/
│  ├─ gdlz/
│  │  ├─ cv/                  # YOLO / 人脸 / SR 实现
│  │  ├─ llm/llama_cpp/       # GGUF / llama.cpp 封装
│  │  ├─ plugin/              # 插件与示例
│  │  └─ trt/                 # TensorRT 实现
│  ├─ test/                   # 各模块测试代码
│  └─ main.cpp                # 当前示例入口
├─ dev-doc/                   # 开发文档草稿
├─ CMakeLists.txt             # 构建脚本
└─ poj.txt                    # 项目笔记 / 临时文档
```

## 依赖说明

从 `CMakeLists.txt` 和测试目标来看，项目目前依赖或预留了以下组件：

- `OpenCV`
- `llama.cpp`
- `nlohmann/json`
- `dylog`
- `ONNX Runtime`
- `LibTorch`
- `TensorRT`
- `CUDA`
- `cuDNN`
- `sd.cpp` / Stable Diffusion 相关库

## 环境要求

当前配置明显偏向以下环境：

- Windows
- Visual Studio 2019 / 2022
- CMake `3.31+`
- C++20
- CUDA 12.x

`CMakeLists.txt` 中还存在一些**本地硬编码路径**，例如：

- `E:/ZeroPlan/C++/Clib`
- `E:/VScode/CUDA`
- `E:/ZeroPlan/C++/Clib/cudnn-8.9-cu12`

这意味着项目目前更适合作者本地环境或相似目录结构下直接使用。如果你准备在另一台机器上构建，建议先统一整理依赖路径。

## 构建方式

### 1. 修改依赖根目录

先检查 `CMakeLists.txt` 中的这一行：

```cmake
set(CLIB_ROOT "E:/ZeroPlan/C++/Clib")
```

将其改为你本机的第三方库根目录。

### 2. 使用 CMake 生成工程

示例：

```powershell
cmake -S . -B build -G "Visual Studio 17 2022"
cmake --build build --config Release
```

如果你的 CUDA、TensorRT、LibTorch、OpenCV 安装路径与当前脚本不一致，还需要同步修改 `CMakeLists.txt` 中对应的 `include` 和 `lib` 路径。

## 当前构建目标

根据 `CMakeLists.txt`，仓库目前会生成或尝试生成以下目标：

- `GeneralDnnLib_Framework_Lib`
- `GeneralDnnLib_Framework_LLM_GGUF_Lib`
- `GeneralDnnLib_Framework_invoke_test`
- `GeneralDnnLib_Torch_Test`
- `GeneralDnnLib_StableDiffusion_Test`
- `GeneralDnnLib_LLM_Test`
- `GeneralDnnLib_TRT_Test`
- `onnx` 测试目标

说明：这些目标依赖较多本地第三方库，并不一定能在“纯净环境”下一次性全部通过。

## 测试入口

测试代码集中在 `src/test/`：

- `src/test/framework/yoloTest.cpp`
- `src/test/invoke/invokeTest.cpp`
- `src/test/llm/llm_class_test.cpp`
- `src/test/llm/llm_test.cpp`
- `src/test/onnx/onnx_test1.cpp`
- `src/test/sd/stable_diffusion_test.cpp`
- `src/test/tf/tensorf_test.cpp`
- `src/test/torch/torch_test.cpp`
- `src/test/trt/trt_test.cpp`
- `src/test/trt/trt_test_2.cpp`

如果你想了解某个模块如何被调用，最直接的方式就是从这些测试文件入手。

## 接口风格示例

以 YOLO 模块为例，当前接口风格大致是：

1. 创建框架对象
2. 使用 `ResourceDirector(...)` 注入资源
3. 调用 `Input(...)`
4. 调用 `Forward(...)`
5. 调用 `PostProcess(...)`
6. 使用 `GetOutput(...)` 取结果

这种写法说明仓库更偏向“链式封装框架”，后续适合继续统一不同模型的调用方式。

## 当前状态

这个项目已经具备以下特征：

- 有较明确的模块划分
- 有公开头文件结构
- 有多种推理后端接入尝试
- 有测试入口与工程脚本

同时也存在一些仍待完善的点：

- 文档还不完整
- 构建脚本耦合本地目录
- 示例配置与模型文件未统一整理
- 跨平台支持暂不明确
- 依赖安装流程还没有完全标准化

## 建议下一步

如果你打算把这个仓库继续完善，建议优先做这几件事：

1. 把所有硬编码路径改成 CMake 变量或环境变量
2. 增加 `config/` 或 `examples/` 目录，统一放示例配置
3. 为 YOLO、LLM、TensorRT 分别补一个最小可运行 demo
4. 在 README 中补充“依赖安装指南”与“模型文件准备方式”
5. 为库目标和测试目标增加更清晰的开关选项

## 说明

- 当前 `README.md` 已按仓库现状重新整理
- 原始 README 存在明显编码问题，已直接重写
- 文档内容以当前代码结构和 `CMakeLists.txt` 为准

如果你愿意，我下一步可以继续帮你补：

- 一个更正式的中英双语 `README`
- `快速开始` 示例代码
- `依赖安装说明` 文档
- `CMakeLists.txt` 去硬编码改造
