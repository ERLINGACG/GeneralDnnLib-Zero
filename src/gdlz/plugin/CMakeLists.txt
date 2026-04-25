cmake_minimum_required(VERSION 3.31)
project(GeneralDnnLib_Zero_plugin_Json)

add_library(GeneralDnnLib_Zero_plugin_Json SHARED ./json/gdlz_json_plugin.cpp)

target_include_directories(GeneralDnnLib_Zero_plugin_Json PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CLIB_ROOT}/json/nlohmann
)

add_library(GeneralDnnLib_Zero_plugin_Json_v2 SHARED
        ./json/gdlz_json_plugin_v2.cpp
        ./example/cpp_to_java.cpp
)
target_include_directories(GeneralDnnLib_Zero_plugin_Json_v2 PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CLIB_ROOT}/json/nlohmann
)

add_library(GeneralDnnLib_cpp_to_java SHARED ./example/cpp_to_java.cpp)
target_include_directories(GeneralDnnLib_cpp_to_java PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CLIB_ROOT}/json/nlohmann
)
