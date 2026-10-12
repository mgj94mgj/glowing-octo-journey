#pragma once

#include "vision/types.hpp"

#include <filesystem>
#include <string>

namespace vision {

class ConfigLoader {
public:
    static AppConfig load(const std::filesystem::path& path);
};

std::filesystem::path default_project_root();

}  // namespace vision
