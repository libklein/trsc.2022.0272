
#ifndef FRVCP_BINDINGS_HELPERS_HPP
#define FRVCP_BINDINGS_HELPERS_HPP

#include <sstream>
#include <string>

template<class T>
std::string ostream_to_string(const T& obj) {
    std::stringstream ss;
    ss << obj;
    return ss.str();
}

static constexpr auto os_str = [](const auto& obj) {
    std::stringstream ss;
    ss << obj;
    return ss.str();
};

#endif
