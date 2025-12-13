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

#function(GeneralDnnLib_Exe TARGET_NAME FLIES_LIST)
#    add_executable(${TARGET_NAME} ${FLIES_LIST})
#    set_include(${TARGET_NAME} opencv)
#endfunction()
#link_directories(
#        ${CLIB_ROOT}/opencv/lib
#)

#add_executable(GeneralDnnLib_Zero src/main.cpp)
#set_include(GeneralDnnLib_Zero ${INCLUDE_PATH})
#set_link_lib(GeneralDnnLib_Zero ${LIBRARY_PATH})
#link_lib(GeneralDnnLib_Zero opencv_world4120)


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
    set_include(${TARGET_NAME} ${INCLUDE_PATHS})
    # 3. 设置链接目录
    set_link_lib(${TARGET_NAME} ${LIBRARY_PATHS})
    # 4. 链接具体库
    link_lib(${TARGET_NAME} ${LIBS})
    IF (NOT "${OUTPUT_DIR}" STREQUAL "")
        set_target_properties(${TARGET_NAME} PROPERTIES
                LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_DIR}
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

task_lib(
        GeneralDnnLib_Framework_Lib            # TARGET_NAME
        "${TEST_FLIES}"                    # FILES_LIST（单文件/多文件都支持）
        "${INCLUDE_PATH}"             # INCLUDE_PATHS（多值）
        "${LIBRARY_PATH}"             # LIBRARY_PATHS（多值）
        "${LIBNAME}"                  # LIBS（多值）
        "gdlz_test_lib"                        # OUTPUT_DIR（可选）
)


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