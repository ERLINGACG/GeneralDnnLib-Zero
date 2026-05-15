cmake_minimum_required(VERSION 3.31)
project(GeneralDnnLib_Zero_ort_llm_subdirectory)

Set(I_CPP_FILES
        ./onnx_rt_llm_framework.cpp
        ./onnx_rt_llm_export.cpp
)

add_library(GeneralDnnLib_Zero_ort_llm SHARED ${I_CPP_FILES})
target_include_directories(GeneralDnnLib_Zero_ort_llm PRIVATE ${I_ORT_INCLUDE})
target_link_libraries(
        GeneralDnnLib_Zero_ort_llm PRIVATE GeneralDnnLib_Zero_ort_core
)
