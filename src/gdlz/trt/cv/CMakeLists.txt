project(GeneralDnnLib_Zero_trt_illm_subdirectory)
set(TRT_CV_CPP_FILES
        ./ocr/trt_cv_ocr_framework.cpp
        ./ocr/trt_cv_ocr_director.cpp
        ./ocr/trt_cv_ocr_processer.cpp

        ./core/trt_cv_core.cpp
        ./core/trt_cv_core_director.cpp
        ./core/private/trt_cv_core_impi.cpp
)
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
set(TRT_CV_INCLUDE_DIRS
        ${CMAKE_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_PATH}
        ${Dylog_INCLUDE}
        ${TensorRT_INCLUDE_PATH}
        ${Nlohmann_JOSN_INCLUDE}
        ${CUDA_INCLUDE_PATH}
)
set(TRT_CV_LIB_DIRS
        ${OpenCV_LIB_PATH}
)
set(TRT_CV_LIBRARIES
        opencv_world4120
)

add_library(GeneralDnnLib_Zero_trt_cv SHARED ${TRT_CV_CPP_FILES})
target_link_libraries(GeneralDnnLib_Zero_trt_cv PUBLIC GeneralDnnLib_Zero_trt_core ${TRT_CV_LIBRARIES})
target_include_directories(GeneralDnnLib_Zero_trt_cv PUBLIC ${TRT_CV_INCLUDE_DIRS})
target_link_directories(GeneralDnnLib_Zero_trt_cv PUBLIC ${TRT_CV_LIB_DIRS})

