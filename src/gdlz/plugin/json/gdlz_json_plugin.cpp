//
// Created by HP on 2026/4/6.
//
#include "./gdlz_json_plugin.h"

#include <iostream>
#include <nlohmann/json.hpp>

#include "../../../../../Clib/log/dylog/include/dylog/logger.h"
#include "gdlz/export.h"



typedef void (*JsonLinkCallBack)(const char* key, const char* value, int level,const char* type);



GDLZ_CORE_API void AllocateLargeDataTest(gdlz::plugin::json::JsonLinkObject& info){
    dylog::DynamicLogger().debug("AllocateLargeDataTest");
    fflush(stdout);

    //不会泄漏内存
    auto max_buffer = std::make_unique<char[]>(1024 * 1024 * 100);
    memset(max_buffer.get(),'C',1024 * 1024 * 100);
    // info.value=std::move(max_buffer);
    // info.valueLength=1024 * 1024 * 100;

}






GDLZ_CORE_API int32_t BuildLinkJsonObject(const char* json_str, gdlz::plugin::json::JsonLinkObject& obj)
{
    try{
        std::cout<<"src:"<<json_str<<std::endl;
        const nlohmann::json j = nlohmann::json::parse(std::string(json_str));
        std::cout<<j.dump(4)<<std::endl;
        obj.json = std::make_unique<nlohmann::json>(j);
    }catch(const nlohmann::json::exception& e){
        dylog::DynamicLogger().error("BuildLinkJsonObject error:{}",e.what());
        return -1;
    }
    return 0;
}

GDLZ_CORE_API void JsonLinkGetToKey(
    const char* key,
    gdlz::plugin::json::JsonLinkObject& obj, JsonLinkCallBack callback
){
    if(!obj.json)
    {
        return;
    }
    dylog::DynamicLogger().setInvokeName("JsonLinkGetToKey").debug("v:{}",std::string((*obj.json)[key].dump(4)).c_str());
    auto k=std::string(key);
    auto v=std::string((*obj.json)[key].dump());

    callback(
        k.c_str(),
        v.c_str(),
        0,
        (*obj.json)[key].type_name()
     );
}

GDLZ_CORE_API void JsonLinkGetToIndex(
    const int index,
    gdlz::plugin::json::JsonLinkObject& obj, JsonLinkCallBack callback
){
    if(!obj.json)
    {
        return;
    }


    callback(
        "",
        std::string((*obj.json)[index].dump()).c_str(),
        0,
        (*obj.json)[index].type_name()
     );
}

auto get_items(nlohmann::json& j,JsonLinkCallBack callback,int level=0)
{
    if (!j.is_object() && !j.is_array()) return;

    for (auto& item :j.items()){

        // std::cout<<item.value().type_name()<<std::endl;
        callback(
            item.key().c_str(),
            item.value().dump().c_str(),
            level,
            item.value().type_name()
        );
        get_items(item.value(),callback,level+1);
    }
}

GDLZ_CORE_API void JsonLinkGetToJava(
    gdlz::plugin::json::JsonLinkObject& obj,
    JsonLinkCallBack callback
){

    get_items(*obj.json,callback);
}
GDLZ_CORE_API void getStr(const char* str){
    dylog::DynamicLogger().debug("getStr:{}",str);
    fflush(stdout);
}
