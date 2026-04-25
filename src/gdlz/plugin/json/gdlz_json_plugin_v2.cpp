//
// Created by HP on 2026/4/9.
//
#include "gdlz_json_plugin_v2.h"

#include <iostream>

int gdlz::plugin::json::v2::BuildJsonData(JsonData& data, const char* json_str)
{
    // std::cout<<"BuildJsonData"<<std::endl;
    try
    {
        // std::cout<<json_str<<std::endl;
        nlohmann::json j = nlohmann::json::parse(json_str);
        data.json_root = std::make_unique<nlohmann::json>(j);
        data.json_value = std::make_unique<std::vector<std::pair<std::string,int>>>();
    } catch (const nlohmann::json::exception& e){
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
int gdlz::plugin::json::v2::PostToKey(JsonData& data, const char* key,int type)
{

    data.json_value->push_back({key,type});
    return 0;
}

void gdlz::plugin::json::v2::At(JsonData& data)
{
    int level=0;
    nlohmann::json* current = data.json_root.get();
    auto value = current->dump();
    for (const auto&[key,type]: *data.json_value)
    {
        if (level<data.json_value->size()) { level++; }
        if (type==1){
            auto i_key = std::string(key);
            current = &current->at(i_key);
            value = current->dump();
        }
        if (type==0){
            auto i_key = std::stoi(key);
            current = &current->at(i_key);
            value = current->dump();
        }
    }
    data.json_value->clear();
    // std::cout<<value<<std::endl;
}

int gdlz::plugin::json::v2::PrintJsonData(JsonData& data)
{
    char* base = reinterpret_cast<char*>(&data);
    std::cout << "=== JsonData memory layout ===" << std::endl;
    std::cout << "sizeof(JsonData): " << sizeof(JsonData) << std::endl;
    std::cout << "alignof(JsonData): " << alignof(JsonData) << std::endl;
    std::cout << "offset of json_root: "
              << (reinterpret_cast<char*>(&data.json_root) - base) << std::endl;
    std::cout << "offset of json_now: "
              << (reinterpret_cast<char*>(&data.json_now) - base) << std::endl;
    std::cout << "offset of json_value: "
              << (reinterpret_cast<char*>(&data.json_value) - base) << std::endl;
    std::cout << "offset of now_level: "
              << (reinterpret_cast<char*>(&data.now_level) - base) << std::endl;
    std::cout << "size of json_root: " << sizeof(data.json_root) << std::endl;
    std::cout << "size of now_level: " << sizeof(data.now_level) << std::endl;
    std::cout << "==============================" << std::endl;
    std::cout<<data.json_root->dump(4)<<std::endl;
    return 0;
}

void gdlz::plugin::json::v2::PrintJsonNodeInfo() {
    std::cout << "=== JsonNodeInfo memory layout ===" << std::endl;
    std::cout << "sizeof(JsonNodeInfo): " << sizeof(JsonNodeInfo) << std::endl;
    std::cout << "alignof(JsonNodeInfo): " << alignof(JsonNodeInfo) << std::endl << std::endl;

    // 打印每个成员的偏移、大小和对齐
#define PRINT_MEMBER(member) \
std::cout << "offsetof(" << #member << "): " << offsetof(JsonNodeInfo, member) \
<< "\tsizeof: " << sizeof(decltype(JsonNodeInfo::member)) \
<< "\talignof: " << alignof(decltype(JsonNodeInfo::member)) \
<< std::endl;

    PRINT_MEMBER(key);
    PRINT_MEMBER(key_len);
    PRINT_MEMBER(value);
    PRINT_MEMBER(value_len);
    PRINT_MEMBER(type);
    PRINT_MEMBER(type_len);
    PRINT_MEMBER(level);

#undef PRINT_MEMBER

    // 额外说明 std::unique_ptr<T> 的大小通常等于 sizeof(T*)
    std::cout << "\nNote: std::unique_ptr<char[]> is a pointer wrapper, size = " << sizeof(std::unique_ptr<char[]>) << std::endl;

}


int gdlz::plugin::json::v2::GetValue(JsonData& data, JsonNodeInfo& info)
{
    int level=0;
    nlohmann::json* current = data.json_root.get();
    auto value = current->dump();
    for (const auto&[key,type]: *data.json_value)
    {
             if (level<data.json_value->size()) { level++; }
             if (type==1){
                 auto i_key = std::string(key);
                 current = &current->at(i_key);
                 // std::cout<<current->dump(4)<<std::endl;
                 value = current->dump();
             }
             if (type==0){
                   auto i_key = std::stoi(key);
                   // std::cout << i_key << std::endl;
                   current = &current->at(i_key);
                   // std::cout<<current->dump(4)<<std::endl;
                   value = current->dump();
             }
    }
    data.json_value->clear();

    info.value = std::make_unique<char[]>(value.size());
    info.value_len = static_cast<int>(value.size());
    memcpy(info.value.get(), value.c_str(), info.value_len);

    info.type = std::make_unique<char[]>(std::string(current->type_name()).size());
    info.type_len = static_cast<int>(std::string(current->type_name()).size());
    memcpy(info.type.get(), std::string(current->type_name()).c_str(), info.type_len);

    return 0;
}

void gdlz::plugin::json::v2::JsonHello()
{
    std::cout<<"JsonHello"<<std::endl;
}

void gdlz::plugin::json::v2::add(int a, int b)
{
    std::cout<<"add:a="<<a<<",b="<<b<<",result="<<a+b<<std::endl;
}

// int gdlz::plugin::json::v2::GetJsonToKey(JsonData& data, const char* key)
// {
//     std::cout<<(*data.json_now)[key].dump(4)<<std::endl;
//     auto j = std::make_unique<nlohmann::json>((*data.json_now)[key]);
//     data.json_now = std::move(j);
//     data.now_level++;
//     return 0;
// }

// int gdlz::plugin::json::v2::GetJsonToKeyCallBack(JsonData& data, const char* key,
//     JsonNodeInfoCallback callback)
// {
//     auto value = (*data.json_now)[key];
//     auto j = std::make_unique<nlohmann::json>(value);
//     data.json_now = std::move(j);
//     data.now_level++;
//
//     auto info = JsonNodeInfo();
//     info.level = data.now_level;
//     info.value = std::make_unique<char[]>(value.dump().size());
//     info.value_len = static_cast<int>(value.dump().size());
//     memcpy(info.value.get(), value.dump().c_str(), info.value_len);
//
//     info.type = std::make_unique<char[]>(std::string(value.type_name()).size());
//     info.type_len = static_cast<int>(std::string(value.type_name()).size());
//     memcpy(info.type.get(), std::string(value.type_name()).c_str(), info.type_len);
//
//     callback(info);
//     return 0;
// }
