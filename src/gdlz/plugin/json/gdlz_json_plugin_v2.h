//
// Created by HP on 2026/4/9.
//

#ifndef GENERALDNNLIB_ZERO_GDLZ_JSON_PLUGIN_V2_H
#define GENERALDNNLIB_ZERO_GDLZ_JSON_PLUGIN_V2_H
#include <memory>
#include <nlohmann/json.hpp>
#include <gdlz/export.h>
namespace gdlz::plugin::json::v2 {



    struct JsonData
    {
        std::unique_ptr<nlohmann::json> json_root{};
        std::unique_ptr<nlohmann::json> json_now{};
        std::unique_ptr<std::vector<std::pair<std::string,int>>> json_value{};
        int now_level = 0;
    };

    struct JsonNodeInfo
    {
        std::unique_ptr<char[]> key{};
        int key_len{};

        std::unique_ptr<char[]> value{};
        int value_len{};

        std::unique_ptr<char[]> type{};
        int type_len{};

        int level = 0;

    };

    typedef void (*JsonNodeInfoCallback)(JsonNodeInfo& info);

    GDLZ_CORE_API int BuildJsonData(JsonData& data, const char* json_str);

    GDLZ_CORE_API int PostToKey(JsonData& data,  const char* key,int type);

    GDLZ_CORE_API void At(JsonData& data);

    GDLZ_CORE_API int PrintJsonData(JsonData& data);

    GDLZ_CORE_API void PrintJsonNodeInfo();

    GDLZ_CORE_API int GetValue(JsonData& data, JsonNodeInfo& info);


    GDLZ_CORE_API void JsonHello();

    GDLZ_CORE_API void add(int a,int b);

    //
    // GDLZ_CORE_API int GetJsonToKeyCallBack(JsonData& data, const char* key, JsonNodeInfoCallback callback);
}
#endif //GENERALDNNLIB_ZERO_GDLZ_JSON_PLUGIN_V2_H