cmake_minimum_required(VERSION 3.31)
project(GeneralDnnLib_Zero_llama_cpp_subdirectory)

set(CPP_FILES
        llm_gguf_resources_director.cpp
        llm_gguf_framework_export.cpp
        llm_gguf_framework.cpp
)
set(GDLZ_INCLUDE_PATH
        ${CMAKE_SOURCE_DIR}/include
        ${Dylog_INCLUDE}
        ${Nlohmann_JOSN_INCLUDE}
        ${Llama_CPP_INCLUDE_PATH}
        ${OpenCV_INCLUDE_PATH}
)
set(GDLZ_LIBS_NAME
        opencv_world4120
        llama
        ggml
)
set(GDLZ_LIB_PATH
        ${OpenCV_LIB_PATH}
        ${Llama_CPP_LIB_PATH}
)
add_library(GeneralDnnLib_Zero_llama_cpp SHARED ${CPP_FILES})
target_include_directories(GeneralDnnLib_Zero_llama_cpp PRIVATE ${GDLZ_INCLUDE_PATH})
target_link_directories(GeneralDnnLib_Zero_llama_cpp PRIVATE ${GDLZ_LIB_PATH})
target_link_libraries(GeneralDnnLib_Zero_llama_cpp PRIVATE ${GDLZ_LIBS_NAME})
