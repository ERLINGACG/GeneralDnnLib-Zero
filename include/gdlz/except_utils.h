//
// Created by HP on 2026/5/16.
//

#ifndef GENERALDNNLIB_ZERO_EXCEPT_UTILS_H
#define GENERALDNNLIB_ZERO_EXCEPT_UTILS_H
#include <excpt.h>
#include <iostream>
#include <ostream>

namespace gdlz::except {
    template <typename F >
    int try_catch(F f){
        #ifdef  WIN32
                __try
                {
                    return f();
                }
                __except(EXCEPTION_EXECUTE_HANDLER){
                    std::cerr << "native crash in RunLLmFramework" << std::endl;
                    return -1001;
                }
        #elif



        #endif

    }
}
#endif //GENERALDNNLIB_ZERO_EXCEPT_UTILS_H