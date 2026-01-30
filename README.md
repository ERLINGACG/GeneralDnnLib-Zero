# GeneralDnnLib_Zero

GeneralDnnLib_Zero 是一个功能丰富的 C++ 深度学习库，提供了多种计算机视觉和自然语言处理任务的统一接口。该库支持多种深度学习框架和模型，包括 YOLO 目标检测、LLM 大语言模型等。

## 功能特性

### 计算机视觉
- **目标检测**：支持 YOLO 系列模型
- **人脸技术**：
  - 人脸检测（Yunet）
  - 人脸识别（ArcFace）
  - 人脸特征提取
  - 人脸记录管理
- **图像超分辨率**：支持多种超分辨率模型

### 自然语言处理
- **大语言模型**：基于 llama.cpp 的 GGUF 模型支持
- **文本生成**：支持生成式 AI 模型

### 模型框架支持
- **OpenCV**：计算机视觉库
- **ONNX Runtime**：跨平台模型推理（计划支持）
- **PyTorch (LibTorch)**：深度学习框架集成（计划支持）
- **Stable Diffusion**：图像生成模型（计划支持）

### 硬件加速
- **CUDA**：NVIDIA GPU 加速
- **AVX2**：CPU 指令集优化

## 项目结构

```
GeneralDnnLib_Zero/
├── include/             # 头文件目录
│   └── gdlz/           # 库头文件
├── src/                # 源代码目录
│   ├── gdlz/           # 核心库实现
│   │   ├── cv/         # 计算机视觉模块
│   │   └── llm/        # 大语言模型模块
│   ├── main.cpp        # 主程序入口
│   └── test/           # 测试代码
├── dev-doc/            # 开发文档（规划中）
├── CMakeLists.txt      # CMake 构建脚本
└── cmake-build-*/      # 构建输出目录
```

## 依赖项

- **OpenCV**：图像处理库
- **llama.cpp**：大语言模型推理
- **ONNX Runtime**：模型推理引擎（计划支持）
- **PyTorch (LibTorch)**：深度学习框架（计划支持）
- **CUDA & cuDNN**：GPU 加速
- **nlohmann/json**：JSON 处理
- **dylog**：日志库

## 构建方法

### 前置条件
- CMake 3.31 或更高版本
- Visual Studio 2019/2022 (Windows)
- CUDA 12.8 + cuDNN 8.9 (可选，用于 GPU 加速)
- 所有依赖库已安装并配置好 CLIB_ROOT 环境变量

### Windows 构建

- 未完成，规划中

### 自定义构建配置

可以通过修改 `CMakeLists.txt` 中的函数来自定义构建行为，包括：
- `task_exe`：生成可执行文件
- `task_lib`：生成库文件
- 各种特定模块的构建函数（如 `task_GeneralDnnLib_Framework_Lib`）


## 模块说明

### 计算机视觉框架

#### YOLO 目标检测
- 支持多种 YOLO 模型
- 提供输入、前向推理、后处理和输出获取的完整流程
- 支持图像和原始字节输入

#### 人脸技术
- **Yunet**：高效的人脸检测模型
- **ArcFace**：准确的人脸识别模型
- **特征提取**：提取人脸特征用于比对
- **工具函数**：提供人脸相关的辅助功能

#### 图像超分辨率
- 支持多种超分辨率模型
- 提供统一的接口进行图像增强

### LLM 框架

- 基于 llama.cpp 的 GGUF 模型支持
- 支持文本生成、对话等功能
- 支持流式输出和批量处理

### Stable Diffusion（计划支持）

- 文本到图像生成
- 图像到图像编辑
- 高度可配置的生成参数

## 配置文件

项目使用 JSON 格式的配置文件来管理各种参数，包括：
- 模型路径
- 推理参数
- 硬件加速设置
- 输出配置

## 性能优化

- **GPU 加速**：通过 CUDA 和 cuDNN 实现 GPU 加速
- **CPU 优化**：使用 AVX2 指令集加速 CPU 推理
- **内存管理**：优化内存使用，减少不必要的内存分配
- **批处理**：支持批量推理，提高吞吐量

## 测试

项目包含丰富的测试代码，位于 `src/test/` 目录下：
- `yoloTest.cpp`：YOLO 目标检测测试
- `llm_class_test.cpp`：LLM 模型测试
- `stable_diffusion_test.cpp`：Stable Diffusion 测试（计划支持）
- `torch_test.cpp`：PyTorch 模型测试
- `onnx_test1.cpp`：ONNX 模型测试

## 开发文档（规划中）

开发文档位于 `dev-doc/` 目录下，包含项目设计、模块划分等信息。



## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题或建议，请通过以下方式联系：
- GitHub Issues：[https://github.com/your-username/GeneralDnnLib_Zero/issues]