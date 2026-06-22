//
// Created by HP on 2026/6/5.
//

#ifndef GENERALDNNLIB_ZERO_TRT_CORE_DIRECTOR_H
#define GENERALDNNLIB_ZERO_TRT_CORE_DIRECTOR_H
#include "trt_core_data.h"
#include "gdlz/export.h"
namespace gdlz::trt::core{

    class GDLZ_CPP_API TensorRTCoreDirector {
        public:
             TensorRTCoreDirector()=default;
             ~TensorRTCoreDirector()=default;

             static int Handle(TensorRTCoreEngine& engine,const char* config_path);
             static int OnnxHandler(TensorRTCoreEngine& engine,const char* config_path);

             template<typename T>
             static int Handle(T& engine,const char* config_path){
                return Handle(engine.getCoreEngine(),config_path);
             };

    };
}
#endif //GENERALDNNLIB_ZERO_TRT_CORE_DIRECTOR_H