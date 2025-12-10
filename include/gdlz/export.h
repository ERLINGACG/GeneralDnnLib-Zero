//
// Created by HP on 2025/12/9.
//

#ifndef EXPORT_H
#define EXPORT_H

#ifdef _WIN32
    #define  GDLZ_CORE_API extern "C" __declspec(dllexport) //Windows跨语言导出宏
    #define  GDLZ_CPP_API            __declspec(dllexport) //Windows跨语言导出宏
#else
    #define GDLZ_CPP_API
    #define GDLZ_CORE_API extern "C"
#endif



#endif //EXPORT_H
