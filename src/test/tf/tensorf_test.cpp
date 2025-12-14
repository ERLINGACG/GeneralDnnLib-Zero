//
// Created by HP on 2025/12/15.
//
extern "C" {
    #include "tensorflow/c/c_api.h"
}
#include <iostream>

int main()
{
    std::cout<<"tensorflow version: "<<TF_Version()<<std::endl;
    return 0;
}
