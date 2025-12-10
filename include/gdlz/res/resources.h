//
// Created by HP on 2025/12/9.
//

#ifndef RESOURCES_H
#define RESOURCES_H
#include <memory>
#include <opencv2/opencv.hpp>
namespace gdlz::resources
{
    // 资源类型
    struct  ResourceBase
    {
        std::unique_ptr<cv::dnn::Net> Net;
        virtual ~ResourceBase()=default;
    };



    template<typename  T,typename R>
    class ResourceDirectorBase
    {
        public:

           virtual ~ResourceDirectorBase()=default;
           virtual R& Hand(T& resource)=0;
    };


} // namespace gdlz::resources
#endif //RESOURCES_H
