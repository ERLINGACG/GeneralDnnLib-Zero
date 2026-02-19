cmake_minimum_required(VERSION 3.31)
project(GeneralDnnLib_Zero)

set(CMAKE_CXX_STANDARD 20)
if(MSVC)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "86")  # 对应3050TI的Ampere架构
    add_definitions(-DUNICODE -D_UNICODE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2 /fp:fast /openmp")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /DNDEBUG /GL")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /utf-8")
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

set(CLIB_ROOT "E:/ZeroPlan/C++/Clib")


function(set_include TARGET_NAME)
    target_include_directories(${TARGET_NAME} PRIVATE ${ARGN})
endfunction()

function(set_link_lib TARGET_NAME)
    target_link_directories(
            ${TARGET_NAME} PRIVATE ${ARGN}
    )
endfunction()

function(link_lib TARGET_NAME)
    target_link_libraries(${TARGET_NAME} PRIVATE ${ARGN})
endfunction()


function(task_exe
        TARGET_NAME       # 参数1：目标名
        FILES_LIST        # 参数2：源文件列表（多值）
        INCLUDE_PATHS     # 参数3：头文件路径列表（多值）
        LIBRARY_PATHS     # 参数4：链接目录列表（多值）
        LIBS              # 参数5：链接库名列表（多值）
        OUTPUT_DIR        # 参数6：输出名（可选）
)
    # 1. 生成可执行文件（FILES_LIST是多值，直接传）
    add_executable(${TARGET_NAME} ${FILES_LIST})
    # 2. 设置头文件路径（用ARGN逻辑传递多值）
    set_include(${TARGET_NAME} ${INCLUDE_PATHS})
    # 3. 设置链接目录
    set_link_lib(${TARGET_NAME} ${LIBRARY_PATHS})
    # 4. 链接具体库
    link_lib(${TARGET_NAME} ${LIBS})
    IF (NOT "${OUTPUT_DIR}" STREQUAL "")
        set_target_properties(${TARGET_NAME} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR}
        )
    ENDIF()

endfunction()

function(task_lib
        TARGET_NAME       # 参数1：目标名
        FILES_LIST        # 参数2：源文件列表（多值）
        INCLUDE_PATHS     # 参数3：头文件路径列表（多值）
        LIBRARY_PATHS     # 参数4：链接目录列表（多值）
        LIBS              # 参数5：链接库名列表（多值）
        OUTPUT_DIR        # 参数6：输出名（可选）
)
    # 1. 生成库文件（FILES_LIST是多值，直接传）
    add_library(${TARGET_NAME} SHARED ${FILES_LIST})
    # 2. 设置头文件路径（用ARGN逻辑传递多值）
    # ============ 新增：仅给当前库生成汇编代码 START ============
    # 开启汇编生成 + 指定汇编输出目录（和你的dll/lib同目录，方便查看）
#    set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_FLAGS "/FA /Fa${OUTPUT_DIR}/asm/")
    # ============ 新增：仅给当前库生成汇编代码 END ============
    set_include(${TARGET_NAME} ${INCLUDE_PATHS})
    # 3. 设置链接目录
    set_link_lib(${TARGET_NAME} ${LIBRARY_PATHS})
    # 4. 链接具体库
    link_lib(${TARGET_NAME} ${LIBS})
    IF (NOT "${OUTPUT_DIR}" STREQUAL "")
        set_target_properties(${TARGET_NAME} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY
                ${OUTPUT_DIR}
                ARCHIVE_OUTPUT_DIRECTORY
                ${OUTPUT_DIR}
        )
    ENDIF()

endfunction()

set(INCLUDE_PATH
        ./include
        ${CLIB_ROOT}/log/dylog/include
        ${CLIB_ROOT}/opencv/include
        ${CLIB_ROOT}/json/nlohmann
)
set(LIBRARY_PATH ${CLIB_ROOT}/opencv/lib cmake-build-release-visual-studio/Release )
set(LIBNAME opencv_world4120)
set(FLIES
        src/main.cpp
)


task_exe(
        GeneralDnnLib_Zero            # TARGET_NAME
        "${FLIES}"                    # FILES_LIST（单文件/多文件都支持）
        "${INCLUDE_PATH}"             # INCLUDE_PATHS（多值）
        "${LIBRARY_PATH}"             # LIBRARY_PATHS（多值）
        "${LIBNAME}"                  # LIBS（多值）
        "test1"                        # OUTPUT_DIR（可选）
)

set(TEST_FLIES
        src/gdlz/cv/framework/yolo/yolo_framework.cpp
        src/gdlz/cv/framework/yolo/yolo_resources_director.cpp
        src/gdlz/cv/framework/sr/sr_framework.cpp
        src/gdlz/cv/framework/sr/sr_resources_director.cpp
#        src/test/framework/yoloTest.cpp
)



task_exe(
        GeneralDnnLib_Yolo_Framework_Test            # TARGET_NAME
        "${TEST_FLIES}"                    # FILES_LIST（单文件/多文件都支持）
        "${INCLUDE_PATH}"             # INCLUDE_PATHS（多值）
        "${LIBRARY_PATH}"             # LIBRARY_PATHS（多值）
        "${LIBNAME}"                  # LIBS（多值）
        "gdlz_test"                        # OUTPUT_DIR（可选）
)



function(task_GeneralDnnLib_Framework_Lib)
    set(CPP_FILES
            src/gdlz/cv/framework/yolo/yolo_framework.cpp
            src/gdlz/cv/framework/yolo/yolo_resources_director.cpp
            src/gdlz/cv/framework/yolo/yolo_framework_export.cpp

            src/gdlz/cv/framework/face/yunet/yunet_resources_director.cpp
            src/gdlz/cv/framework/face/yunet/yunet_framework.cpp
            src/gdlz/cv/framework/face/yunet/yunet_framework_export.cpp

            src/gdlz/cv/framework/face/base/feature_dnn_resources_director.cpp

            src/gdlz/cv/framework/face/arcface/arcface_framework_export.cpp
            src/gdlz/cv/framework/face/arcface/arcface_framework.cpp

            src/gdlz/cv/framework/face/utils/face_utils.cpp

            src/gdlz/cv/framework/face/record/record_examlpe.cpp


            src/gdlz/cv/framework/sr/sr_framework.cpp
            src/gdlz/cv/framework/sr/sr_resources_director.cpp

            src/gdlz/llm/llama_cpp/llm_gguf_resources_director.cpp
            src/gdlz/llm/llama_cpp/llm_gguf_framework_export.cpp
            src/gdlz/llm/llama_cpp/llm_gguf_framework.cpp
    )
    set(GDLZ_INCLUDE_PATH
            ./include
            ${CLIB_ROOT}/log/dylog/include
            ${CLIB_ROOT}/opencv/include
            ${CLIB_ROOT}/json/nlohmann
            ${CLIB_ROOT}/llama.cpp/include
    )
    set(GDLZ_LIB_PATH
            ${CLIB_ROOT}/opencv/lib
            ${CLIB_ROOT}/llama.cpp/lib
    )
    set(GDLZ_LIBS_NAME
            opencv_world4120
            llama
    )

    task_lib(
            GeneralDnnLib_Framework_Lib            # TARGET_NAME
            "${CPP_FILES}"                    # FILES_LIST（单文件/多文件都支持）
            "${GDLZ_INCLUDE_PATH}"             # INCLUDE_PATHS（多值）
            "${GDLZ_LIB_PATH}"             # LIBRARY_PATHS（多值）
            "${GDLZ_LIBS_NAME}"                  # LIBS（多值）
            "gdlz_test_lib"                         # OUTPUT_DIR（可选）
    )
endfunction()

function(task_invoke_test)
    set(TEST_INVOKE_FLIES
            src/test/invoke/invokeTest.cpp
    )
    set(TEST_INVOKE_LIBS_NAME
            GeneralDnnLib_Framework_Lib
            opencv_world4120
    )
    set(TEST_INVOKE_LIBS_PATH
            ${CLIB_ROOT}/opencv/lib
            cmake-build-release-visual-studio/Release
    )


    task_exe(
            GeneralDnnLib_Framework_invoke_test            # TARGET_NAME
            "${TEST_INVOKE_FLIES}"                    # FILES_LIST（单文件/多文件都支持）
            "${INCLUDE_PATH}"             # INCLUDE_PATHS（多值）
            "${TEST_INVOKE_LIBS_PATH}"             # LIBRARY_PATHS（多值）
            "${TEST_INVOKE_LIBS_NAME}"                  # LIBS（多值）
            "gdlz_test"                        # OUTPUT_DIR（可选）
    )
endfunction()

function(task_onnx_test)
    set(ONNX_T_FLIES
            src/test/onnx/onnx_test1.cpp
    )

    set(ONNX_T_LIBS_NAME
            onnxruntime
            opencv_world4120
    )
    set(ONNX_T_LIBS_PATH
            ${CLIB_ROOT}/opencv/lib
            ${CLIB_ROOT}/onnxruntime-win-x64-gpu-1.20.1/lib
    )
    set(ONNX_T_INCLUDE_PATH
            ./include
            ${CLIB_ROOT}/log/dylog/include
            ${CLIB_ROOT}/opencv/include
            ${CLIB_ROOT}/json/nlohmann
            ${CLIB_ROOT}/onnxruntime-win-x64-gpu-1.20.1/include
    )
    task_exe(
            GeneralDnnLib_Onnx_Test            # TARGET_NAME
            "${ONNX_T_FLIES}"                    # FILES_LIST（单文件/多文件都支持）
            "${ONNX_T_INCLUDE_PATH}"             # INCLUDE_PATHS（多值）
            "${ONNX_T_LIBS_PATH}"             # LIBRARY_PATHS（多值）
            "${ONNX_T_LIBS_NAME}"                  # LIBS（多值）
            "gdlz_test"                        # OUTPUT_DIR（可选）
    )
endfunction()

function(task_tensorf_test)
    set(TENSORF_T_FLIES
            src/test/tf/tensorf_test.cpp
    )

    set(TENSORF_T_LIBS_NAME
            onnxruntime
            opencv_world4120
            tensorflow
    )
    set(TENSORF_T_LIBS_PATH
            ${CLIB_ROOT}/opencv/lib
            ${CLIB_ROOT}/onnxruntime-win-x64-gpu-1.20.1/lib
            ${CLIB_ROOT}/tf/lib
    )
    set(TENSORF_T_INCLUDE_PATH
            ./include
            ${CLIB_ROOT}/tf/include
            ${CLIB_ROOT}/log/dylog/include
            ${CLIB_ROOT}/opencv/include
            ${CLIB_ROOT}/json/nlohmann
            ${CLIB_ROOT}/onnxruntime-win-x64-gpu-1.20.1/include
    )
    task_exe(
            GeneralDnnLib_TensorF_Test            # TARGET_NAME
            "${TENSORF_T_FLIES}"                    # FILES_LIST（单文件/多文件都支持）
            "${TENSORF_T_INCLUDE_PATH}"             # INCLUDE_PATHS（多值）
            "${TENSORF_T_LIBS_PATH}"             # LIBRARY_PATHS（多值）
            "${TENSORF_T_LIBS_NAME}"                  # LIBS（多值）
            "gdlz_test"                        # OUTPUT_DIR（可选）
    )


endfunction()

function(task_torch_test)
    # ========== 1. 核心：指定CUDA架构（3050Ti=sm_86，必须优先设置） ==========
    set(TORCH_CUDA_ARCH_LIST "8.6" CACHE STRING "CUDA architectures for LibTorch")
    set(CMAKE_CUDA_ARCHITECTURES "86")  # 兜底兼容VS编译
    set(CMAKE_CXX_STANDARD 17)          # LibTorch要求C++17及以上
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    # ========== 2. 源文件/库名配置（补全CUDA依赖） ==========
    set(TORCH_T_FLIES
            src/test/torch/torch_test.cpp
    )
    set(TORCH_T_LIBS_NAME
            opencv_world4120
            # LibTorch CUDA核心库（必须按此顺序）
            torch.lib          # 主库（放开注释）
            torch_cpu.lib
            torch_cuda.lib
            c10.lib
            c10_cuda.lib
            # CUDA/cuDNN系统库（手动补全，避免链接缺失）
#            cudart64_12.lib    # CUDA 12.8 Runtime
#            cudnn64_8.lib      # cuDNN 8.9（需提前下载）
#            cublas64_12.lib    # CUDA BLAS（可选，提升张量计算性能）
    )

    # ========== 3. 库路径配置（添加CUDA/cuDNN路径） ==========
    set(TORCH_T_LIBS_PATH
            ${CLIB_ROOT}/opencv/lib
            ${CLIB_ROOT}/onnxruntime-win-x64-gpu-1.20.1/lib
            ${CLIB_ROOT}/libtorch/lib
            E:/VScode/CUDA/lib/x64          # 你的CUDA 12.8 lib路径
            E:/ZeroPlan/C++/Clib/cudnn-8.9-cu12/lib/x64  # cuDNN路径（替换为你的实际路径）
    )

    # ========== 4. 查找LibTorch（强制绑定CUDA版本） ==========
    find_package(Torch REQUIRED PATHS ${CLIB_ROOT}/libtorch NO_DEFAULT_PATH)
    if (NOT Torch_FOUND)
        message(FATAL_ERROR "LibTorch not found! Check CLIB_ROOT/libtorch path")
    endif()
    # 打印关键信息（验证CUDA配置）
    message(STATUS "===== LibTorch CUDA Config =====")
    message(STATUS "Torch Version: ${Torch_VERSION}")
    message(STATUS "Torch CUDA Available: ${TORCH_CUDA_AVAILABLE}")
    message(STATUS "Torch Libraries: ${TORCH_LIBRARIES}")
    message(STATUS "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}")

    # ========== 5. 头文件路径（补全CUDA/cuDNN头文件） ==========
    set(TORCH_T_INCLUDE_PATH
            ./include
            ${CLIB_ROOT}/libtorch/include
            ${CLIB_ROOT}/libtorch/include/torch/csrc/api/include  # LibTorch C++ API头文件（易缺失）
            ${CLIB_ROOT}/log/dylog/include
            ${CLIB_ROOT}/opencv/include
            ${CLIB_ROOT}/json/nlohmann
            ${CLIB_ROOT}/onnxruntime-win-x64-gpu-1.20.1/include
            E:/VScode/CUDA/include          # CUDA 12.8头文件路径
            E:/ZeroPlan/C++/Clib/cudnn-8.9-cu12/include  # cuDNN头文件路径
    )

    # ========== 6. 生成可执行文件 ==========
    task_exe(
            GeneralDnnLib_Torch_Test
            "${TORCH_T_FLIES}"
            "${TORCH_T_INCLUDE_PATH}"
            "${TORCH_T_LIBS_PATH}"
            "${TORCH_T_LIBS_NAME}"
            "gdlz_test"
    )

#    # ========== 7. 关键：拷贝LibTorch CUDA所有DLL到输出目录 ==========
#    add_custom_command(TARGET GeneralDnnLib_Torch_Test POST_BUILD
#            # 拷贝LibTorch核心DLL
#            COMMAND ${CMAKE_COMMAND} -E copy_if_different
#            ${CLIB_ROOT}/libtorch/lib/torch.dll
#            ${CLIB_ROOT}/libtorch/lib/torch_cpu.dll
#            ${CLIB_ROOT}/libtorch/lib/torch_cuda.dll
#            ${CLIB_ROOT}/libtorch/lib/c10.dll
#            ${CLIB_ROOT}/libtorch/lib/c10_cuda.dll
#            # 拷贝CUDA系统DLL
#            E:/VScode/CUDA/bin/cudart64_12.dll
#            E:/VScode/CUDA/bin/cublas64_12.dll
#            # 拷贝cuDNN DLL
#            E:/ZeroPlan/C++/Clib/cudnn-8.9-cu12/bin/cudnn64_8.dll
#            # 输出目录
#            $<TARGET_FILE_DIR:GeneralDnnLib_Torch_Test>
#    )

    # ========== 8. 强制启用CUDA编译定义（绕过LibTorch检测bug） ==========
    target_compile_definitions(GeneralDnnLib_Torch_Test PRIVATE
            USE_CUDA=1
            TORCH_USE_CUDA=1
            C10_USE_CUDA=1
    )

    # ========== 9. 链接Torch自动识别的库（兜底） ==========
    target_link_libraries(GeneralDnnLib_Torch_Test PRIVATE ${TORCH_LIBRARIES})
endfunction()

function(task_StableDiffusion_test)
    set(SD_T_FLIES
            src/test/sd/stable_diffusion_test.cpp
    )
    set(SD_T_INCLUDE_PATH
            ./include
            ${CLIB_ROOT}/sd.cpp/install/include
            ${CLIB_ROOT}/libtorch/include
            ${CLIB_ROOT}/libtorch/include/torch/csrc/api/include  # LibTorch C++ API头文件（易缺失）
            ${CLIB_ROOT}/log/dylog/include
            ${CLIB_ROOT}/opencv/include
            ${CLIB_ROOT}/json/nlohmann
            ${CLIB_ROOT}/onnxruntime-win-x64-gpu-1.20.1/include
            E:/VScode/CUDA/include          # CUDA 12.8头文件路径
            E:/ZeroPlan/C++/Clib/cudnn-8.9-cu12/include  # cuDNN头文件路径
    )
    set(SD_T_LIBS_PATH
            ${CLIB_ROOT}/sd.cpp/install/lib
            ${CLIB_ROOT}/opencv/lib
            ${CLIB_ROOT}/onnxruntime-win-x64-gpu-1.20.1/lib
            ${CLIB_ROOT}/libtorch/lib
            E:/VScode/CUDA/lib/x64          # 你的CUDA 12.8 lib路径
            E:/ZeroPlan/C++/Clib/cudnn-8.9-cu12/lib/x64  # cuDNN路径（替换为你的实际路径）
    )
    set(SD_T_LIBS_NAME
            opencv_world4120
            ggml-base
            ggml-cpu
            ggml-cuda
            ggml
            stable-diffusion
            cuda
            cudart
            cudadevrt
            cublas
            cublasLt
            cusparse
            curand
    )


    task_exe(
            GeneralDnnLib_StableDiffusion_Test            # TARGET_NAME
            "${SD_T_FLIES}"                     # FILES_LIST（单文件/多文件都支持）
            "${SD_T_INCLUDE_PATH}"             # INCLUDE_PATHS（多值）
            "${SD_T_LIBS_PATH}"             # LIBRARY_PATHS（多值）
            "${SD_T_LIBS_NAME}"                  # LIBS（多值）
            "gdlz_test"                        # OUTPUT_DIR（可选）
    )
endfunction()

function(task_llm_test)


    set(LLM_T_FLIES
            src/test/llm/llm_class_test.cpp
    )
    set(LLM_T_INCLUDE_PATH
            ./include
            ${CLIB_ROOT}/llama.cpp/include
            ${CLIB_ROOT}/sd.cpp/install/include
            ${CLIB_ROOT}/libtorch/include
            ${CLIB_ROOT}/libtorch/include/torch/csrc/api/include  # LibTorch C++ API头文件（易缺失）
            ${CLIB_ROOT}/log/dylog/include
            ${CLIB_ROOT}/opencv/include
            ${CLIB_ROOT}/json/nlohmann
            ${CLIB_ROOT}/onnxruntime-win-x64-gpu-1.20.1/include
            E:/VScode/CUDA/include          # CUDA 12.8头文件路径
            E:/ZeroPlan/C++/Clib/cudnn-8.9-cu12/include  # cuDNN头文件路径
    )
    set(LLM_T_LIBS_PATH
            ${CLIB_ROOT}/llama.cpp/lib
            ${CLIB_ROOT}/sd.cpp/install/lib
            ${CLIB_ROOT}/opencv/lib
            ${CLIB_ROOT}/onnxruntime-win-x64-gpu-1.20.1/lib
            ${CLIB_ROOT}/libtorch/lib
            E:/VScode/CUDA/lib/x64          # 你的CUDA 12.8 lib路径
            E:/ZeroPlan/C++/Clib/cudnn-8.9-cu12/lib/x64  # cuDNN路径（替换为你的实际路径）
    )
    set(LLM_T_LIBS_NAME
            opencv_world4120
            ggml-base
            ggml-cpu
            ggml-cuda
            ggml
            llama
            cuda
            cudart
            cudadevrt
            cublas
            cublasLt
            cusparse
            curand
    )


    task_exe(
            GeneralDnnLib_LLM_Test            # TARGET_NAME
            "${LLM_T_FLIES}"                     # FILES_LIST（单文件/多文件都支持）
            "${LLM_T_INCLUDE_PATH}"             # INCLUDE_PATHS（多值）
            "${LLM_T_LIBS_PATH}"             # LIBRARY_PATHS（多值）
            "${LLM_T_LIBS_NAME}"                  # LIBS（多值）
            "gdlz_test"                        # OUTPUT_DIR（可选）
    )

endfunction()

function(task_trt_test)
  set(TRT_FILE
            src/test/trt/trt_test.cpp
  )

  set(TRT_INCLUDE
         ./include
          ${CLIB_ROOT}/opencv/include
          ${CLIB_ROOT}/TensorRT-10.12.0.36/include
          E:/VScode/CUDA/include
  )
  set(TRT_LIBS_PATH
          ${CLIB_ROOT}/TensorRT-10.12.0.36/lib
          E:/VScode/CUDA/lib/x64

  )

  set(TRT_LIBS_NAME
          nvinfer_10
          cudart.lib
  )

  task_exe(
            GeneralDnnLib_TRT_Test            # TARGET_NAME
            "${TRT_FILE}"                     # FILES_LIST（单文件/多文件都支持）
            "${TRT_INCLUDE}"             # INCLUDE_PATHS（多值）
            "${TRT_LIBS_PATH}"             # LIBRARY_PATHS（多值）
            "${TRT_LIBS_NAME}"                  # LIBS（多值）
            "gdlz_test"                        # OUTPUT_DIR（可选）
  )

endfunction()
task_invoke_test()
task_onnx_test()
task_torch_test()
#task_tensorf_test() //已否决
task_GeneralDnnLib_Framework_Lib()
task_StableDiffusion_test()
task_llm_test()
task_trt_test()