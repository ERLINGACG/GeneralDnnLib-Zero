cmake_minimum_required(VERSION 3.31)
project(GeneralDnnLib_Zero_trt_subdirectory)

Set(I_CPP_FILES
    ./rag/trt_emb_framework.cpp
    ./rag/trt_emb_director.cpp
    ./rag/trt_emb_export.cpp
)
set(I_TRT_INCLUDE
        ${CMAKE_SOURCE_DIR}/include
        ${CLIB_ROOT}/opencv/include
        ${CLIB_ROOT}/log/dylog/include
        ${CLIB_ROOT}/TensorRT-10.12.0.36/include
        ${CLIB_ROOT}/json/nlohmann
        E:/VScode/CUDA/include
)
set(I_TRT_LIBS_PATH
        ${CLIB_ROOT}/TensorRT-10.12.0.36/lib
        E:/VScode/CUDA/lib/x64

)

set(I_TRT_LIBS_NAME
        nvinfer_10
        cudart.lib
)

#add_library(GeneralDnnLib_Zero_trt_emdb SHARED ${I_CPP_FILES})
#target_include_directories(GeneralDnnLib_Zero_trt_emdb PRIVATE ${I_TRT_INCLUDE})
#target_link_directories(GeneralDnnLib_Zero_trt_emdb PRIVATE ${I_TRT_LIBS_PATH})
#target_link_libraries(GeneralDnnLib_Zero_trt_emdb PRIVATE ${I_TRT_LIBS_NAME})


function(build_trt_lib lib_name cpp_files include_dirs lib_dirs lib_names)
    add_library(${lib_name} SHARED ${cpp_files})
    target_include_directories(${lib_name} PRIVATE ${include_dirs})
    target_link_directories(${lib_name} PRIVATE ${lib_dirs})
    target_link_libraries(${lib_name} PRIVATE ${lib_names})
endfunction()


build_trt_lib(GeneralDnnLib_Zero_trt_emb "${I_CPP_FILES}" "${I_TRT_INCLUDE}" "${I_TRT_LIBS_PATH}" "${I_TRT_LIBS_NAME}")



