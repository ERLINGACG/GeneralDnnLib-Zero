//
// Created by HP on 2026/4/6.
//

#ifndef GENERALDNNLIB_ZERO_GDLZ_JSON_PLUGIN_H
#define GENERALDNNLIB_ZERO_GDLZ_JSON_PLUGIN_H
#include <iostream>
#include <memory>
#include <string>
#include <nlohmann/json.hpp>


namespace gdlz::plugin::json{



    struct JsonLinkObject{
        std::unique_ptr<nlohmann::json> json{};
    };



}
#endif //GENERALDNNLIB_ZERO_GDLZ_JSON_PLUGIN_H