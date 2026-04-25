cmake_minimum_required(VERSION 3.31)
project(GeneralDnnLib_Zero_llama_cpp_subdirectory)

set(CPP_FILES
        ./llm_gguf_resources_director.cpp
        ./llm_gguf_framework_export.cpp
        ./llm_gguf_framework.cpp
)
set(GDLZ_INCLUDE_PATH
        ${CMAKE_SOURCE_DIR}/include
        ${CLIB_ROOT}/log/dylog/include
        ${CLIB_ROOT}/opencv/include
        ${CLIB_ROOT}/json/nlohmann/include
        ${CLIB_ROOT}/llama.cpp/include
)
set(GDLZ_LIBS_NAME
        opencv_world4120
        llama
        ggml
)
set(GDLZ_LIB_PATH
        ${CLIB_ROOT}/opencv/lib
        ${CLIB_ROOT}/llama.cpp/lib
)
add_library(GeneralDnnLib_Zero_llama_cpp SHARED ${CPP_FILES})
target_include_directories(GeneralDnnLib_Zero_llama_cpp PRIVATE ${GDLZ_INCLUDE_PATH})
target_link_directories(GeneralDnnLib_Zero_llama_cpp PRIVATE ${GDLZ_LIB_PATH})
target_link_libraries(GeneralDnnLib_Zero_llama_cpp PRIVATE ${GDLZ_LIBS_NAME})
