cmake_minimum_required(VERSION 3.31)
project(GeneralDnnLib_Zero_trt_subdirectory CXX CUDA)

Set(I_CPP_FILES
    ./rag/trt_emb_framework.cpp
    ./rag/trt_emb_director.cpp
    ./rag/trt_emb_export.cpp
)
set(CORE_CPP_FILES
        ./core/trt_core_framework.cpp
        ./core/trt_core_director.cpp
        ./core/trt_core_export.cpp

        core/cuda/op/trt_cuda_post.cpp
        core/cuda/op/trt_cuda_get.cpp
        core/cuda/op/trt_cuda_memory.cpp
        core/cuda/op/trt_cuda_address.cpp
)
set(I_TRT_INCLUDE
        ${CMAKE_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${Dylog_INCLUDE}
        ${TensorRT_INCLUDE_PATH}
        ${Nlohmann_JOSN_INCLUDE}
        E:/VScode/CUDA/include
)
set(I_TRT_LIBS_PATH
        ${TensorRT_LIB_PATH}
        E:/VScode/CUDA/lib/x64
)

set(I_TRT_LIBS_NAME
        nvinfer_10
        cudart.lib
        nvonnxparser_10
)



function(build_trt_lib lib_name cpp_files include_dirs lib_dirs lib_names)
    add_library(${lib_name} SHARED ${cpp_files})
    target_include_directories(${lib_name} PRIVATE ${include_dirs})
    target_link_directories(${lib_name} PRIVATE ${lib_dirs})
    target_link_libraries(${lib_name} PRIVATE ${lib_names})
endfunction()


function(build_trt_core_lib lib_name cpp_files include_dirs lib_dirs lib_names)
    add_library(${lib_name} SHARED ${cpp_files})
    target_include_directories(${lib_name} PUBLIC ${include_dirs})
    target_link_directories(${lib_name} PUBLIC ${lib_dirs})
    target_link_libraries(${lib_name} PUBLIC ${lib_names})
endfunction()

build_trt_lib(GeneralDnnLib_Zero_trt_emb
        "${I_CPP_FILES}"
        "${I_TRT_INCLUDE}"
        "${I_TRT_LIBS_PATH}"
        "${I_TRT_LIBS_NAME}"
)
build_trt_core_lib(GeneralDnnLib_Zero_trt_core
        "${CORE_CPP_FILES}"
        "${I_TRT_INCLUDE}"
        "${I_TRT_LIBS_PATH}"
        "${I_TRT_LIBS_NAME}")
add_subdirectory(illm)


