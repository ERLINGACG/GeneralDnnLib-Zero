cmake_minimum_required(VERSION 3.31)
project(GeneralDnnLib_Zero_ort_subdirectory)


Set(I_CPP_FILES
        ./core/onnx_rt_core_framework.cpp
        ./core/onnx_rt_core_director.cpp
        ./core/onnx_rt_core_export.cpp
)

set(I_ORT_INCLUDE
        ${CMAKE_SOURCE_DIR}/include
        ${CLIB_ROOT}/opencv/include
        ${CLIB_ROOT}/log/dylog/include
        ${CLIB_ROOT}/json/nlohmann
        ${CLIB_ROOT}/onnxruntime-win-x64-gpu-1.20.1/include
)

set(I_ORT_LIBS_PATH
        E:/VScode/CUDA/lib/x64
        ${CLIB_ROOT}/onnxruntime-win-x64-gpu-1.20.1/lib
)
set(I_ORT_LIBS_NAME
        onnxruntime
#        onnxruntime_providers_cuda
)

add_library(GeneralDnnLib_Zero_ort_core_framework SHARED ${I_CPP_FILES})
target_include_directories(GeneralDnnLib_Zero_ort_core_framework PRIVATE ${I_ORT_INCLUDE})
target_link_directories(GeneralDnnLib_Zero_ort_core_framework PRIVATE ${I_ORT_LIBS_PATH})
target_link_libraries(GeneralDnnLib_Zero_ort_core_framework PRIVATE ${I_ORT_LIBS_NAME})
