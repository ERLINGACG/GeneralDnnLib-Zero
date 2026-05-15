cmake_minimum_required(VERSION 3.31)
project(GeneralDnnLib_Zero_sr_subdirectory)

set(SR_U_CPP_FILES
        sr_utils.cpp
)
set(SR_U_INCLUDE
        ${CMAKE_SOURCE_DIR}/include
        ${CLIB_ROOT}/opencv/include
)
set(SR_U_LIBS_PATH
        ${CLIB_ROOT}/opencv/lib
)

set(SR_U_LIBS_NAME
        opencv_world4120
)
add_executable(GeneralDnnLib_Zero_SR ${SR_U_CPP_FILES})
target_include_directories(GeneralDnnLib_Zero_SR PRIVATE ${SR_U_INCLUDE})
target_link_directories(GeneralDnnLib_Zero_SR PRIVATE ${SR_U_LIBS_PATH})
target_link_libraries(GeneralDnnLib_Zero_SR PRIVATE ${SR_U_LIBS_NAME})
