//
// Created by HP on 2026/6/7.
//

#ifndef GENERALDNNLIB_ZERO_TRT_ERR_H
#define GENERALDNNLIB_ZERO_TRT_ERR_H
namespace gdlz::trt::code{

    namespace operation{
         enum  TensoRTCode : int32_t {
             SUCCESS =  0,
             FAILURE = -1,
             RT_FAILURE = -2,
         };
    }

    namespace address{
        enum  TensorRTCode : int32_t {
            ADDRESS_NULL = -1001,
        };
    }

    namespace ctx{
        enum  TensorRTCode : int32_t {
            CONTEXT_NULL            = -2001,
            SET_INPUT_SHAPE_FAILURE = -2002,
            BING_ADDRESS_FAILURE    = -2003,
        };
    }

    namespace shap{
        enum  TensorRTCode : int32_t {
            SHAPE_DIMS_ZERO =   -3001,
            SET_SHAPE_FAILURE = -3002,
            SHAPE_DIMS_FAILURE = -3003,
        };
    }

    namespace cuda
    {
        enum TensorRTCode : int32_t {
            MALLOC_FAILED = -4001,
            AYS_FAILED = -4002,
            SYN_FAILED = -4003,
        };
    }

    namespace data
    {
        enum  TensorRTCode : int32_t{
           DATA_NULL = -5001,
           MODEL_DATA_EMPTY = -5002,
        };
    }

    namespace config{
        enum  TensorRTCode : int32_t {
            CONFIG_NOT_FLIED =   -6001,
            READ_CONFIG_FAILED = -6002,
        };
    }

}
#endif //GENERALDNNLIB_ZERO_TRT_ERR_H