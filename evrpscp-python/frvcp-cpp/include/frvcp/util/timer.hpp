#pragma once

#include <chrono>
#include <cstdlib>
#include <sys/time.h>
#include <type_traits>

template<class unit> class Timer {
    using ClockType = std::chrono::steady_clock;

    ClockType::time_point start_t;
    ClockType::time_point end_t;
    ClockType::duration _total_duration;
    // Timer should be steady.
    static_assert(ClockType::is_steady);
    // Timer has at least the precision required
    static_assert(ClockType::period::den >= unit::period::den);

public:
    Timer()
        : _total_duration(0) {};

    void start() { start_t = ClockType::now(); }

    void stop() {
        end_t = ClockType::now();
        _total_duration += end_t - start_t;
    }

    void reset() {
        _total_duration = unit(0);
        end_t           = start_t;
    }

    unit duration() const { return std::chrono::duration_cast<unit>(_total_duration); }

    unit lastRound() const { return std::chrono::duration_cast<unit>(end_t - start_t); }

    friend std::ostream& operator<<(std::ostream& os, const Timer& r) {
        std::string suffix = "";
        if constexpr (std::is_same<unit, std::chrono::microseconds>::value) {
            suffix = "µs";
        } else if (std::is_same<unit, std::chrono::seconds>::value) {
            suffix = "s";
        } else if (std::is_same<unit, std::chrono::milliseconds>::value) {
            suffix = "ms";
        } else if (std::is_same<unit, std::chrono::nanoseconds>::value) {
            suffix = "ns";
        }
        os << r.duration().count() << suffix;
        return os;
    };
};

using STimer = Timer<std::chrono::seconds>;
using MTimer = Timer<std::chrono::milliseconds>;
using UTimer = Timer<std::chrono::microseconds>;
using NTimer = Timer<std::chrono::nanoseconds>;
