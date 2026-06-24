//
// Created by HP on 2026/6/5.
//

#ifndef GENERALDNNLIB_ZERO_TRT_CORE_FRAMEWORK_H
#define GENERALDNNLIB_ZERO_TRT_CORE_FRAMEWORK_H
#include <dylog/logger.h>

#include "trt_core_data.h"
#include "cuda/trt_cuda_data.h"
#include "gdlz/export.h"
namespace gdlz::trt::core{




    class GDLZ_CPP_API TensorRTCoreFramework {


        public:
         static int CreateEngine (TensorRTCoreEngine& engine);

         static int HotBuildEngine(TensorRTCoreEngine& engine,const char* config_path);

         static int InitShapeInfo(TensorBuilderEngineInfo& info);

         static int RegisterShapeInfo(const TensorRtBuilderShapeInfo& shape,TensorBuilderEngineInfo& info);

        static int  GetShapeInfoDetail(TensorBuilderEngineInfo& info);

         static int BuildEngine(TensorRTCoreEngine& engine,const char* config_path,TensorBuilderEngineInfo& info);

         static int GetEngineInfo(TensorRTCoreEngine& engine);

         static int CreateCtx(TensorRTCoreEngine& engine ,TensorRTCoreCtx& ctx);

         static int SetLayerShape(TensorRTCoreCtx& ctx,const char* name,const int64_t* shape,int64_t shape_size);

         static int Forward(TensorRTCoreCtx& ctx);

         static int TestRun(TensorRTCoreEngine& engine);


         template<typename T>
         static int CreateEngine(T& engine){
             return CreateEngine(engine.getCoreEngine());
         }

         template<typename T,typename C>
         static int CreateCtx(T& engine,C& ctx){
             return CreateCtx(engine.getCoreEngine(),ctx.getCoreCtx());
         }

         template<typename T>
         static int Forward(T& ctx){
             return Forward(ctx.getCoreCtx());
         }

         template<typename T>
         static int SetLayerShape(T& ctx,const char* name,const int64_t* shape,int64_t shape_size){
             return TensorRTCoreFramework::SetLayerShape(ctx.getCoreCtx(),name,shape,shape_size);
         }

         template<typename T>
         static int BuildEngine(T& engine,const char* config_path,TensorBuilderEngineInfo& info){
             return BuildEngine(engine.getCoreEngine(),config_path,info);
         }
    };
}
#endif //GENERALDNNLIB_ZERO_TRT_CORE_FRAMEWORK_H