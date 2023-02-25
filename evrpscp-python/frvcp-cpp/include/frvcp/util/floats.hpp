
#ifndef FRVCP_UTIL_HPP
#define FRVCP_UTIL_HPP

#include <cmath>

namespace frvcp {

    inline bool approx_eq(double x, double y, double _eps = EPS) {
        return std::abs(x - y) <= std::max(_eps * std::abs(x), _eps);
    }

    inline bool approx_lt(double x, double y, double _eps = EPS) { return (x < y) || approx_eq(x, y, _eps); }

    inline bool approx_gt(double x, double y, double _eps = EPS) { return (x > y) || approx_eq(x, y, _eps); }

    inline bool certainly_lt(double x, double y, double _eps = EPS) { return x < y && !approx_eq(x, y, _eps); }

    inline bool certainly_gt(double x, double y, double _eps = EPS) {
        return x > y && !approx_eq(x, y, _eps);
        return x - y > _eps;
    }

}

#endif // FRVCP_UTIL_HPP
