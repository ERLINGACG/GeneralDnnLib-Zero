//
// Created by HP on 2026/4/10.
//

#include "cpp_to_java.h"

gdlz::plugin::example::CppToJava* gdlz::plugin::example::CreateCppToJava()
{
    return new CppToJava();
}

void gdlz::plugin::example::PrintCppToJava(const CppToJava* cpp_to_java)
{
    std::cout<<cpp_to_java->cpp_code<<std::endl;
}

void gdlz::plugin::example::SetStruct(CppTestStruct& cpp_test_struct)
{
    cpp_test_struct.a = 100;
    cpp_test_struct.b = 200;
    cpp_test_struct.name = "test";
}

void gdlz::plugin::example::PrintCppStr(const CppToJava* cpp_to_java, const char* str)
{
    cpp_to_java->iprint(str);
}

void gdlz::plugin::example::PrintCppTestStruct(CppTestStruct& cpp_test_struct)
{
    std::cout<<"name: "<<cpp_test_struct.name<<std::endl;
    std::cout<<"a: "<<cpp_test_struct.a<<std::endl;
    std::cout<<"b: "<<cpp_test_struct.b<<std::endl;
}

void gdlz::plugin::example::SetStruct2(CppTestStruct2& cpp_test_struct2)
{
    strcpy(cpp_test_struct2.name, "test2");
    cpp_test_struct2.a = 300;
}

void gdlz::plugin::example::PrintCppTestStruct2(const CppTestStruct2& cpp_test_struct2)
{
    std::cout<<"name: "<<cpp_test_struct2.name<<std::endl;
    std::cout<<"a: "<<cpp_test_struct2.a<<std::endl;
}

void gdlz::plugin::example::DestroyCppToJava(const CppToJava* cpp_to_java)
{
    delete cpp_to_java;
}

