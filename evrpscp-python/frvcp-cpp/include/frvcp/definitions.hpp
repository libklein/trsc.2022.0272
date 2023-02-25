
#ifndef FRVCP_DEFINITIONS_HPP
#define FRVCP_DEFINITIONS_HPP

#include <bitset>
#include <cstddef>
#include <limits>

namespace frvcp {
    static constexpr size_t DEFAULT_ALLOCATOR_POOL_SIZE = 10000000UL;
    static constexpr double PERIOD_LENGTH               = 30.0;

    using tour_id                                = size_t;
    using charger_id                             = size_t;
    constexpr static charger_id MAX_NUM_CHARGERS = std::numeric_limits<charger_id>::max();
    constexpr static tour_id MAX_NUM_TOURS       = 32;
    using tour_set_t                             = std::bitset<MAX_NUM_TOURS>;

    /// Represents infeasible SoC
    static constexpr double INVALID_SOC = std::numeric_limits<double>::lowest();
    /// Represents cost of unreachable SoC levels
    static constexpr double COST_OF_UNREACHABLE_SOC = std::numeric_limits<double>::max();

    using Period = unsigned int;

    inline static constexpr double to_time(Period periods) { return periods * PERIOD_LENGTH; }

    constexpr static double MAX_SLOPE = 100.0;
    constexpr static double MIN_SLOPE = EPS;
    constexpr static double PRECISION = []() constexpr {
        if (EPS == 0.1)
            return 10.0;
        if (EPS == 0.01)
            return 100.;
        if (EPS == 0.001)
            return 1000.;
        if (EPS == 0.0001)
            return 10000.;
        if (EPS == 0.00001)
            return 100000.;
        if (EPS == 0.000001)
            return 1000000.;
        return 0.;
    }();
    static_assert(EPS * PRECISION == 1.0);
}

#endif // FRVCP_DEFINITIONS_HPP
