#include <iostream>
#include <opencv2/opencv.hpp>
#include <windows.h>
#include <nlohmann/json.hpp>
// 定义全局类，利用构造函数在main前执行SetDllDirectory
// class DllPathSetter {
// public:
//     DllPathSetter() {
//         const wchar_t* dllPath = L"E:/ZeroPlan/C++/Clib/opencv/bin";
//         SetDllDirectory(dllPath); // 比main早执行，覆盖DLL搜索路径
//     }
// };
// // 全局对象，程序启动时自动调用构造函数
// DllPathSetter g_dllPathSetter;
int main()
{
    auto img=cv::imread("bus.jpg");
    cv::imshow("bus.jpg", img);
    cv::waitKey(0);
    return 0;
}