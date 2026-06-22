cmake_minimum_required(VERSION 3.31)
project(GeneralDnnLib_Zero_trt_illm_subdirectory)

set(ILLM_CU_FILES
        ./sample/cu/trt_illm_temp.cu
)

set(ILLM_CPP_FILES
        ./trt_illm_framework.cpp
        ./trt_illm_director.cpp
        ./trt_illm_export.cpp

        ./sample/sp/trt_illm_sample.cpp

)

add_library(GeneralDnnLib_Zero_trt_illm SHARED ${ILLM_CPP_FILES} ${ILLM_CU_FILES})
target_include_directories(GeneralDnnLib_Zero_trt_illm PUBLIC ${I_TRT_INCLUDE})
target_link_libraries(
        GeneralDnnLib_Zero_trt_illm PUBLIC GeneralDnnLib_Zero_trt_core
)
