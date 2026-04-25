//
// Created by HP on 2026/4/10.
//

#ifndef GENERALDNNLIB_ZERO_CPP_TO_JAVA_H
#define GENERALDNNLIB_ZERO_CPP_TO_JAVA_H
#include <iostream>
#include <ostream>

#include "gdlz/export.h"
namespace gdlz::plugin::example
{
    struct CppTestStruct
    {
        const char* name;
        int a;
        int b;
    };

    struct CppTestStruct2
    {
        char name[100];
        int a;
    };

    class GDLZ_CPP_API CppToJava
    {

    public:
        std::string cpp_code="hello world java world";

        CppToJava(){ std::cout<<"CppToJava()"<<std::endl;};
        ~CppToJava(){ std::cout<<"~CppToJava()"<<std::endl;};
        void  iprint(const char* str)  const { std::cout<<str<<std::endl; };
    };

    GDLZ_CORE_API CppToJava* CreateCppToJava();

    GDLZ_CORE_API void PrintCppToJava(const CppToJava* cpp_to_java);

    GDLZ_CORE_API void SetStruct(CppTestStruct& cpp_test_struct);

    GDLZ_CORE_API void PrintCppStr(const CppToJava* cpp_to_java,const char* str);

    GDLZ_CORE_API void PrintCppTestStruct(CppTestStruct& cpp_test_struct);

    GDLZ_CORE_API void SetStruct2(CppTestStruct2& cpp_test_struct2);

    GDLZ_CORE_API void PrintCppTestStruct2(const CppTestStruct2& cpp_test_struct2);

    GDLZ_CORE_API void DestroyCppToJava(const CppToJava* cpp_to_java);
}
#endif //GENERALDNNLIB_ZERO_CPP_TO_JAVA_H