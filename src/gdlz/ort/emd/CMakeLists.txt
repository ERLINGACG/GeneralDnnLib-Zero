cmake_minimum_required(VERSION 3.31)
project(GeneralDnnLib_Zero_ort_rag_subdirectory)

Set(I_CPP_FILES
       ./onnx_rt_emd_framework.cpp
       ./onnx_rt_emd_director.cpp
       ./onnx_rt_emd_export.cpp
)

add_library(GeneralDnnLib_Zero_ort_emd SHARED ${I_CPP_FILES})
target_include_directories(GeneralDnnLib_Zero_ort_emd PUBLIC ${I_ORT_INCLUDE})
target_link_libraries(
        GeneralDnnLib_Zero_ort_emd PUBLIC GeneralDnnLib_Zero_ort_core
)
