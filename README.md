# GeneralDnnLib-Zero

## 项目定位

- **一个库，三个后端**：TensorRT / ONNX Runtime / llama.cpp，统一接口风格
- **跨语言直达**：`extern "C"` 导出层 + [Java-DNN-Zero](https://github.com/ERLINGACG/Java-DNN-Zero) 绑定，一套 DLL 同时服务 C++ 和 Java 调用方
- **从引擎构建到逐 token 推理**：覆盖 ONNX → Engine 构建、Prefill/Decode 循环、KV Cache 管理、CUDA 内存分配
- **工程实践优先**：不是 Demo 堆砌，是真实可用的 DLL 项目

## 架构概览

```
┌───────────────────────────────────────────────┐
│                  调用方                        │
│   C++ (GDLZ_CPP_API)  │  Java (GDLZ_CORE_API) │
├───────────────────────────────────────────────┤
│            Export 层 (extern "C")              │
├──────────┬──────────┬──────────┬──────────────┤
│  TRT     │  ORT     │ llama    │  CV          │
│  ├ core  │  ├ core  │  .cpp    │  ├ YOLO      │
│  ├ illm  │  ├ llm   │          │  ├ YuNet     │
│  └ rag   │  └ emd   │          │  ├ ArcFace   │
│          │          │          │  └ SR        │
├──────────┴──────────┴──────────┴──────────────┤
│   CUDA Memory / Stream / Post / Get / Sync    │
├───────────────────────────────────────────────┤
│  TensorRT    │  ONNX Runtime  │  llama.cpp    │
│  nvinfer     │  onnxruntime   │  ggml         │
│  CUDA 12.x   │                │               │
└───────────────────────────────────────────────┘
```

每个子模块遵循统一的 **Director → Framework → Export** 三层模式：

| 层 | 职责 |
|------|------|
| **Director** | 配置加载、资源编排、JSON 解析 |
| **Framework** | 核心业务逻辑（引擎构建、推理、后处理） |
| **Export** | `extern "C"` 接口，消除 name mangling，供 Java / C# 调用 |

## 模块清单

### TensorRT 模块 (`gdlz::trt`)

| 子模块 | DLL | 能力 |
|--------|-----|------|
| `core` | `GeneralDnnLib_Zero_trt_core` | ONNX → Engine 构建（HotBuild + 通用构建）、引擎反序列化、执行上下文管理、CUDA 内存/H2D/D2H 封装 |
| `illm` | `GeneralDnnLib_Zero_trt_illm` | LLM 自回归推理：Prefill/Decode 分离、KV Cache 零拷贝管理、手写 Argmax CUDA Kernel、温度采样 |
| `rag` | `GeneralDnnLib_Zero_trt_emb` | RAG Embedding 推理：动态 shape 输入、FP16/FP32 支持、Mean Pooling（基于 attention_mask） |

### ONNX Runtime 模块 (`gdlz::ort`)

| 子模块 | DLL | 能力 |
|--------|-----|------|
| `core` | `GeneralDnnLib_Zero_ort_core` | ONNX Runtime 引擎封装、Session 管理、配置加载 |
| `llm` | `GeneralDnnLib_Zero_ort_llm` | ORT 后端 LLM 推理：Prefill/Decode、RoPE、采样器、多 Batch 生成 |
| `emd` | `GeneralDnnLib_Zero_ort_emd` | ORT 后端 Embedding 推理：Shape 设置、Forward、结果提取 |

### llama.cpp 模块 (`gdlz::llm`)

| 模块 | DLL | 能力 |
|------|-----|------|
| `llama_cpp` | `GeneralDnnLib_Zero_llama_cpp` | GGUF 模型加载与推理、批处理、流式输出、Embedding 提取，异步采样与自回归 |

### 计算机视觉模块 (`gdlz::cv`)

#### 注意: 该模块暂时已经过时，并且长久没更新了，使用起来可能遇到一些未知的问题。

| 子模块 | 能力 |
|--------|------|
| YOLO | 目标检测：资源加载、前处理、推理、后处理、输出提取 |
| YuNet | 人脸检测 |
| ArcFace | 人脸特征提取 / 识别 |
| SR | 图像超分辨率 |

### 插件与工具

| 模块 | DLL | 说明 |
|------|-----|------|
| plugin/Json | `GeneralDnnLib_Zero_plugin_Json` / `_v2` | JSON 解析插件，可供 Java 调用 |
| utils/sr | `GeneralDnnLib_Zero_SR` | 超分图像处理工具（独立可执行文件） |

## 跨语言 API 设计

项目提供两层导出宏（定义于 `include/gdlz/export.h`）：

```cpp
// C++ 调用方：标准 C++ ABI
#define GDLZ_CPP_API  __declspec(dllexport)

// 跨语言调用方：消除 name mangling
#define GDLZ_CORE_API extern "C" __declspec(dllexport)
```

以 TRT-ILLM 为例，Java 端加载同一个 DLL 的典型调用链：

```
Java (FFM) → LoadLibrary("GeneralDnnLib_Zero_trt_illm.dll")
           → CreateTensorRTIllmFramework(config_path)
           → CreateTensorRTIllmEngine(framework)
           → RegisterInput(framework, "input_ids", "INP")
           → RegisterKvMapping(framework, "past_kv.0.key", "present_kv.0.key")
           → AutoInputIds(ctx, token_ids, len)
           → PrefillDefault(ctx)
           → DecodeDefault(ctx)   // 自回归循环
           → DestroyTensorRTIllmFramework(framework)
```

完整 Java 绑定实现见 [Java-DNN-Zero](https://github.com/ERLINGACG/Java-DNN-Zero)。

## 目录结构

```text
GeneralDnnLib-Zero/
├── CMakeLists.txt                  # 根构建脚本（配置中心）
├── README.md
├── include/gdlz/
│   ├── export.h                    # 跨语言导出宏
│   ├── res/                        # 资源管理基类
│   ├── trt/
│   │   ├── core/                   # TRT Core：引擎构建、CUDA 内存、C API
│   │   │   ├── trt_core_framework.h
│   │   │   ├── trt_core_director.h
│   │   │   ├── trt_core_export.h
│   │   │   ├── trt_core_data.h
│   │   │   ├── trt_code.h          # 错误码命名空间
│   │   │   └── cuda/op/            # CUDA Post/Get/Memory/Address
│   │   ├── illm/                   # TRT LLM：Prefill/Decode、采样、导出
│   │   │   ├── trt_illm_framework.h
│   │   │   ├── trt_illm_export.h
│   │   │   └── sample/
│   │   │       ├── cu/             # CUDA Kernel（Argmax）
│   │   │       └── sp/             # 采样器
│   │   └── rag/                    # TRT RAG：Embedding 推理
│   │       ├── trt_emb_framework.h
│   │       ├── trt_emb_director.h
│   │       └── trt_emb_export.h
│   ├── ort/
│   │   ├── data/                   # 共享数据结构
│   │   ├── core/                   # ORT Core：Session 管理
│   │   ├── llm/                    # ORT LLM：推理循环、采样器
│   │   └── emd/                    # ORT Embedding
│   ├── llama/framework/            # llama.cpp GGUF 封装
│   └── cv/framework/
│       ├── yolo/                   # YOLO 目标检测
│       ├── sr/                     # 图像超分
│       └── face/
│           ├── yunet/              # 人脸检测
│           ├── arcface/            # 人脸识别
│           ├── base/               # 人脸 DNN 基类
│           └── utils/              # 人脸工具函数
├── src/gdlz/
│   ├── trt/
│   │   ├── CMakeLists.txt          # 构建 trt_emb + trt_core
│   │   ├── core/                   # TRT Core 实现
│   │   ├── illm/                   # TRT LLM 实现（~400 行）
│   │   │   ├── CMakeLists.txt
│   │   │   └── sample/cu/          # ArgmaxKernelFP32 实现
│   │   └── rag/                    # TRT RAG 实现（~600 行，含 Mean Pooling）
│   ├── ort/
│   │   ├── CMakeLists.txt
│   │   ├── core/
│   │   ├── llm/
│   │   └── emd/
│   ├── llama_cpp/
│   │   └── CMakeLists.txt
│   ├── plugin/
│   │   ├── CMakeLists.txt
│   │   ├── json/                   # JSON 插件
│   │   └── example/                # C++↔Java 互操作示例
│   └── cv/                         # YOLO / 人脸 / SR 实现
├── src/utils/sr/                   # 超分工具（独立 exe）
└── src/test/                       # 各模块测试代码
    ├── trt/trt_test.cpp            # 原始 TensorRT API 测试（YOLO 推理）
    ├── trt/trt_test_2.cpp          # Embedding 推理测试
    ├── llm/                        # LLM 测试
    ├── onnx/                       # ORT 测试
    ├── torch/                      # LibTorch 测试
    ├── sd/                         # Stable Diffusion 测试
    └── tf/                         # TensorFlow 测试
```

## 构建系统架构

根 `CMakeLists.txt` 作为**配置中心**，通过 `CACHE` 变量集中管理所有第三方库路径：

```cmake
set(TensorRT_INCLUDE_PATH  CACHE PATH "Path to TensorRT include directory")
set(TensorRT_LIB_PATH      CACHE PATH "Path to TensorRT library directory")
set(CUDA_INCLUDE_PATH      CACHE PATH "Path to CUDA include directory")
# ... OpenCV, llama.cpp, ONNX Runtime, LibTorch, nlohmann/json, dylog
```

自定义 DSL 函数封装重复模板：

```cmake
task_lib(TARGET_NAME  FILES  INCLUDE_PATHS  LIBRARY_PATHS  LIBS  OUTPUT_DIR)
task_exe(TARGET_NAME  FILES  INCLUDE_PATHS  LIBRARY_PATHS  LIBS  OUTPUT_DIR)
```

### DLL 依赖传递设计

```
trt_emb (PRIVATE)          → 独立 DLL，不暴露依赖
trt_core (PUBLIC)          → 公开头文件和链接，供 illm 继承
  └─ trt_illm (PUBLIC依赖) → 自动继承 trt_core 的全部 include/lib

ort_core (PUBLIC)          → 公开依赖
  ├─ ort_llm (PUBLIC依赖)  → 继承 ort_core
  └─ ort_emd (PUBLIC依赖)  → 继承 ort_core

plugin/Json (PRIVATE)      → 独立
llama_cpp (PRIVATE)        → 独立
```

关键：`target_link_libraries(trt_illm PUBLIC GeneralDnnLib_Zero_trt_core)` 这一行，让 `trt_illm` 自动继承 `trt_core` 的 TensorRT、CUDA、nlohmann/json 所有头文件路径和链接库——子模块 CMakeLists 只需声明自己的源文件。

### 编译器配置

| 选项 | 说明 |
|------|------|
| C++20 | 语言标准 |
| CUDA arch 86 | RTX 3050 Ti (Ampere) |
| `/arch:AVX2` | CPU 向量化 |
| `/fp:fast` | 浮点快速模式 |
| `/openmp` | CPU 并行 |
| `/GL` + `/LTCG` | Release 全程序优化 + 链接时代码生成 |
| `/utf-8` | 源码编码 |

## 依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| **TensorRT** | 10.12.0.36 | TRT 引擎构建与推理 |
| **CUDA** | 12.8 | GPU 计算 + 内存管理 |
| **ONNX Runtime** | 1.20.1 (GPU) | ORT 后端推理 |
| **llama.cpp** | 8210 (2cd20b72e) | GGUF 模型推理 |
| **OpenCV** | 4.12.0 | 图像处理 |
| **nlohmann/json** | — | JSON 配置解析 |
| **dylog** | — | 统一日志 |
| **LibTorch** | 2.x (CUDA) | Torch 测试 |
| **sd.cpp** | — | Stable Diffusion 测试 |

## 环境要求

- Windows 10/11
- Visual Studio 2022
- CMake 3.31+
- C++20
- CUDA 12.x（GPU 模块）
- RTX 3050 Ti 或同代 Ampere GPU（其他架构需修改 `CMAKE_CUDA_ARCHITECTURES`）

## 快速开始

### 1. 准备第三方库

确保本地有类似如下目录结构（或修改 `CMakeLists.txt` 中的 `CLIB_ROOT`）：

```text
E:/ZeroPlan/C++/Clib/
├── TensorRT-10.12.0.36/
│   ├── include/
│   └── lib/
├── onnxruntime-win-x64-gpu-1.20.1/
├── opencv/
├── llama.cpp/
├── json/nlohmann/
├── log/dylog/
└── libtorch/          # 可选
```

CUDA 路径：`E:/VScode/CUDA/`

### 2. 修改本地路径

编辑 `CMakeLists.txt`，修改以下变量为你的实际路径：

```cmake
set(CLIB_ROOT "你的第三方库根目录")
set(CUDA_INCLUDE_PATH "你的CUDA/include路径")
set(CUDA_LIB_PATH     "你的CUDA/lib/x64路径")
```

### 3. 构建

```powershell
cmake -S . -B build -G "Visual Studio 17 2022"
cmake --build build --config Release
```

构建产物在 `build/` 对应子目录下，包括：

- `GeneralDnnLib_Zero_trt_core.dll` + `.lib`
- `GeneralDnnLib_Zero_trt_illm.dll` + `.lib`
- `GeneralDnnLib_Zero_trt_emb.dll` + `.lib`
- `GeneralDnnLib_Zero_ort_core.dll` + `.lib`
- `GeneralDnnLib_Zero_ort_llm.dll` + `.lib`
- `GeneralDnnLib_Zero_ort_emd.dll` + `.lib`
- `GeneralDnnLib_Zero_llama_cpp.dll` + `.lib`
- `GeneralDnnLib_Zero_plugin_Json.dll` + `.lib`

### 4. 运行测试（暂未完善）

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -DBUILD_TO_TEST=ON
cmake --build build --config Release
```

测试可执行文件输出至 `gdlz_test/`。

## 接口调用示例

以 TRT-ILLM 为例，完整推理流程：

```cpp
// 1. 创建框架，加载 ONNX 模型配置
auto* fw = CreateTensorRTIllmFramework("config.json");

// 2. 构建 TensorRT Engine（或直接加载已有 engine）
CreateTensorRTIllmEngine(fw);

// 3. 注册输入张量
RegisterInput(fw, "input_ids", "INP");       // 输入 token
RegisterInput(fw, "attention_mask", "MSK");  // 注意力掩码
RegisterInput(fw, "position_ids", "POS");    // 位置编码
RegisterLogits(fw, "logits", "LOG");         // 输出 logits

// 4. 注册 KV Cache 映射（28 层，每层 key + value）
RegisterKvMapping(fw, "past_key_values.0.key", "present_key_values.0.key");
RegisterKvMapping(fw, "past_key_values.0.value", "present_key_values.0.value");
// ... 共 28 × 2 = 56 个映射

// 5. 创建推理上下文
TensorRTIllmCtx ctx;
CreateTensorRTIllmCtx(fw, ctx);

// 6. 设置采样参数
TensorRTIllmSampleParams params{/*top_k, top_p, temperature*/};
SetSampler(ctx, params);

// 7. Prefill：一次性注入全部输入
int64_t input_ids[] = {1, 2, 3, 4, 5};  // tokenized prompt
AutoInputIds(ctx, input_ids, 5);
AutoPastKvCacheDefault(ctx, dims, dims_len);
AutoPresentKvCacheDefault(ctx);
AutoLogits(ctx);
PrefillDefault(ctx);  // 执行 + 采样，得到 next_token

// 8. Decode：逐 token 自回归
while (ctx.next_token != eos_token_id) {
    DecodeDefault(ctx);  // 指针交换 KV Cache + 单 token 推理
}

// 9. 清理
DestroyTensorRTIllmFramework(fw);
```

## 当前状态

**已完成：**

- TRT / ORT / llama.cpp 三个后端的完整封装
- TRT-ILLM 的 Prefill/Decode 自回归推理循环
- TRT-RAG 的 Mean Pooling Embedding 推理
- CUDA 内存管理、H2D/D2H 的类型安全封装
- 手写 Argmax CUDA Kernel
- `extern "C"` 跨语言导出层 + Java 绑定
- CMake DSL 构建体系（7 个 DLL 的依赖传递）

**待完善：**

- 跨平台支持（当前仅 Windows）
- 硬编码路径去耦合（`CLIB_ROOT`、CUDA 路径等改为环境变量）
- 统一示例配置与模型文件
- 文档与使用教程（博客系列撰写中）
- ORT-LLM 模块的完整度提升

## 相关项目

- [Java-DNN-Zero](https://github.com/ERLINGACG/Java-DNN-Zero) — 本项目的 Java FFM 绑定层，Android 兼容

## 开发者留言:

readme.md是deepseek-r4-flash写的，所以...