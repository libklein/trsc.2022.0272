
#ifndef FRVCP_DEBUG_HPP
#define FRVCP_DEBUG_HPP

#include <frvcp/models/fwd.hpp>
#include <frvcp/time_expanded_network.hpp>
#include <vector>

#if defined(WARN_EXTENSIVE_SAFETY_CHECK) || defined(THROW_EXTENSIVE_SAFETY_CHECKS)
#define ENABLE_EXTENSIVE_SAFETY_CHECK true
#else
#define ENABLE_EXTENSIVE_SAFETY_CHECK false
#endif

namespace frvcp::util {

    void handle_extensive_check_failure(std::string_view message) {
#if defined WARN_EXTENSIVE_SAFETY_CHECK
        std::cerr << message << std::endl;
#else
        throw std::runtime_error(message.data());
#endif
    }

}

#endif // FRVCP_DEBUG_HPP
