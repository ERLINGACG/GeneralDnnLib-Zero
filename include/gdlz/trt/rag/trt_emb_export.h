//
// Created by HP on 2026/4/19.
//

#ifndef GENERALDNNLIB_ZERO_TRT_EMB_EXPORT_H
#define GENERALDNNLIB_ZERO_TRT_EMB_EXPORT_H
#include "trt_emb_framework.h"
#include "gdlz/export.h"

namespace gdlz::trt::rag {
      GDLZ_CORE_API TrtEmbFramework* CreateTrtEmbFramework(const char* config_path);
      GDLZ_CORE_API void DestroyTrtEmbFramework(const TrtEmbFramework* framework);

      GDLZ_CORE_API void GetEngineInfo(const TrtEmbFramework* framework);
      GDLZ_CORE_API int CreateCtx(const TrtEmbFramework* framework, TrtEmbCtx& ctx);
      GDLZ_CORE_API int SetDims(TrtEmbCtx& ctx, const char* name, const int* input, int len);
      GDLZ_CORE_API int GetDims(const TrtEmbFramework* framework, TrtEmbCtx& ctx, const char* name);

      GDLZ_CORE_API int PostCuda(
          const TrtEmbFramework* framework,
          const TrtEmbCtx& ctx,
          const char* name,
          const void* input, int len, int isMask
      );

      GDLZ_CORE_API int ForwardAsync(const TrtEmbFramework* framework, TrtEmbCtx& ctx);
      GDLZ_CORE_API int Forward     (const TrtEmbFramework* framework, TrtEmbCtx& ctx);
      GDLZ_CORE_API int Synchronize(const TrtEmbCtx& ctx);


      GDLZ_CORE_API int GetEmbedding(
          const TrtEmbFramework* framework,
          const TrtEmbCtx& ctx, TrtEmbData& embedding,
          const char* name
      );
      GDLZ_CORE_API int GetPooledEmbedding(
          const TrtEmbFramework* framework,
          const TrtEmbCtx& ctx, TrtEmbData& embedding,
          const char* hidden_state_name,
          const char* attention_mask_name
      );

      GDLZ_CORE_API int ClearBindings(TrtEmbCtx& ctx);


}
#endif //GENERALDNNLIB_ZERO_TRT_EMB_EXPORT_H
