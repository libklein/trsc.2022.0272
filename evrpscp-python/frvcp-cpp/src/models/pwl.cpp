#include "frvcp/models/pwl.hpp"
#include "frvcp/models/pwl_util.hpp"
#include "frvcp/util/floats.hpp"

#include <algorithm>
#include <boost/functional/hash.hpp>
#include <cassert>
#include <fmt/format.h>
#include <ranges>
#ifdef ENABLE_SAFETY_CHECKS
#include <iostream>
#endif

namespace {
    [[nodiscard]] bool optimized_approx_equal_axes(
        const frvcp::models::PWLFunction::breakpoint_t& lhs, const frvcp::models::PWLFunction::breakpoint_t& rhs) {
        return (frvcp::approx_eq(lhs.domain, rhs.domain) && frvcp::approx_eq(lhs.image, rhs.image));
    }

    [[nodiscard]] bool optimized_approx_equal_slopes(
        const frvcp::models::PWLFunction::breakpoint_t& lhs, const frvcp::models::PWLFunction::breakpoint_t& rhs) {
        return (frvcp::approx_eq(lhs.slope * (rhs.domain - lhs.domain), rhs.slope * (rhs.domain - lhs.domain)));
    }
}

namespace frvcp::models {
    void recompute_slopes(frvcp::models::PWLFunction::bp_container_t& breakpoints) {
        if (breakpoints.size() < 2) {
            return;
        }
        auto prev = breakpoints.begin();
        for (auto cur = std::next(prev); cur != breakpoints.end(); prev = cur++) {
            cur->slope = compute_slope(*prev, *cur);
        }
    }

    PWLFunction::PWLFunction(PWLFunction::bp_container_t breakpoints)
        : _segments(std::move(breakpoints)) {
        auto [min_elem, max_elem] = std::minmax_element(
            _segments.begin(), _segments.end(), [](const auto& lhs, const auto& rhs) { return lhs.image < rhs.image; });
        _minimum_image = min_elem->image;
        _maximum_image = max_elem->image;
    }

    PWLFunction::bp_container_t& PWLFunction::_get_segments() { return _segments; }
    const PWLFunction::bp_container_t& PWLFunction::_get_segments() const { return _segments; }

    const PWLFunction::bp_container_t& PWLFunction::getBreakpoints() const { return _segments; }

    double PWLFunction::getImageUpperBound() const { return _maximum_image; }

    double PWLFunction::getUpperBound() const { return _get_segments().back().domain; }

    std::strong_ordering PWLFunction::operator<=>(const PWLFunction& rhs) const {
        assert(getLowerBound() == rhs.getLowerBound() && getUpperBound() == rhs.getUpperBound());

        auto lhs_seg     = _get_segments().begin();
        auto rhs_seg     = rhs._get_segments().begin();
        double lhs_value = getImageLowerBound(), rhs_value = rhs.getImageLowerBound();
        for (; lhs_seg != _get_segments().end() && rhs_seg != _get_segments().end();) {
            if (lhs_value < rhs_value) {
                return std::strong_ordering::less;
            } else if (lhs_value > rhs_value) {
                return std::strong_ordering::greater;
            }

            if (lhs_seg->domain <= rhs_seg->domain) {
                ++lhs_seg;
                lhs_value = lhs_seg->image;
                rhs_value = rhs(lhs_seg->domain);
            } else {
                ++rhs_seg;
                lhs_value = (*this)(rhs_seg->domain);
                rhs_value = rhs_seg->image;
            }
        }

        if (lhs_seg == _get_segments().end() && rhs_seg == _get_segments().end()) {
            return std::strong_ordering::equivalent;
        }
        return (lhs_seg == _get_segments().end()) ? std::strong_ordering::less : std::strong_ordering::greater;
    }

    bool PWLFunction::operator==(const PWLFunction& rhs) const { return _segments == rhs._segments; }

    double PWLFunction::operator()(double x) const { return value(x); }

    double PWLFunction::value(double domain_value) const {
        if (domain_value >= getUpperBound())
            return getImageUpperBound();

        // First segment that is an lower bound on the domain value
        auto seg = std::find_if(_get_segments().rbegin(), _get_segments().rend(),
            [domain_value](const auto& segment) { return segment.domain <= domain_value; });
        // Use std::prev here as seg is a reverse iterator! This will correspond to the first segment for which domain_value
        // < segment.domain
        double val = seg->image + (domain_value - seg->domain) * std::prev(seg)->slope;
        return val;
    }

    double PWLFunction::inverse(double image_value) const {
        if (image_value >= getImageUpperBound())
            return getUpperBound();

        auto seg = std::find_if(_get_segments().rbegin(), _get_segments().rend(),
            [image_value](const auto& segment) { return segment.image <= image_value; });
        // Use std::prev here as seg is a reverse iterator! This will correspond to the first segment for which image_value <
        // segment.image
        double slope = std::prev(seg)->slope;
        if (slope == 0.0)
            return seg->domain;
        double val = seg->domain + (image_value - seg->image) / slope;
        return val;
    }

    double PWLFunction::slope(double x) const {
        if (x == getUpperBound()) {
            return _get_segments().back().slope;
        }
        auto seg = std::find_if(
            _get_segments().rbegin(), _get_segments().rend(), [x](const auto& segment) { return segment.domain <= x; });
        // Use std::prev here as seg is a reverse iterator!
        return std::prev(seg)->slope;
    }

    double PWLFunction::slope_at_inverse(double y) const {
        if (y > getImageUpperBound())
            return 0.0;
        else if (y == getImageUpperBound())
            return _get_segments().back().slope;

        auto seg = std::find_if(
            _get_segments().rbegin(), _get_segments().rend(), [y](const auto& segment) { return segment.image <= y; });

        return std::prev(seg)->slope;
    }

    double PWLFunction::getImageLowerBound() const { return _minimum_image; }

    double PWLFunction::getLowerBound() const { return _get_segments().front().domain; }

    PWLFunction::iterator PWLFunction::begin() { return _get_segments().begin(); }
    PWLFunction::iterator PWLFunction::end() { return _get_segments().end(); }
    PWLFunction::const_iterator PWLFunction::begin() const { return getBreakpoints().begin(); }
    PWLFunction::const_iterator PWLFunction::end() const { return getBreakpoints().end(); }

    std::ostream& operator<<(std::ostream& os, const PWLFunction& function) {
        auto prev = function.begin();
        for (auto cur = std::next(prev); cur != function.end(); prev = cur++) {
            os << "[" << prev->domain << ", " << cur->domain << ")"
               << " -> [" << prev->image << ", " << cur->image << ")@" << cur->slope << '\n';
        }
        return os;
    }

    Segment::Segment(double domain_value, double image_value)
        : domain(domain_value)
        , image(image_value)
        , slope(PWLFunction::LB_SLOPE) { }

    Segment::Segment(double domain_value, double image_value, double slope_value)
        : domain(domain_value)
        , image(image_value)
        , slope(slope_value) { }

    std::strong_ordering Segment::operator<=>(const Segment& other) const {
        if (domain < other.domain) {
            return std::strong_ordering::less;
        } else if (domain > other.domain) {
            return std::strong_ordering::greater;
        } else {
            return std::strong_ordering::equivalent;
        }
    }

    std::ostream& operator<<(std::ostream& os, const PWLFunction::breakpoint_t& segment) {
        os << "[" << segment.domain << ", " << segment.image << ", " << segment.slope << "]";
        return os;
    }

    void optimize_breakpoint_sequence(PWLFunction::bp_container_t& breakpoints) {
        /**
         * Optimize BPs aggressively, potentially deleting more BPs that allowed (i.e., when LB == UB)
         * Then fix such problems afterwards
         */
        if (breakpoints.size() < 2)
            return;

        auto prev_valid_segment = breakpoints.end();
        auto last_valid_segment = breakpoints.begin();
        auto next_segment       = last_valid_segment;
        while (++next_segment != breakpoints.end()) {
            bool bps_have_equal_position = optimized_approx_equal_axes(*last_valid_segment, *next_segment);
            bool bps_have_equal_slope    = optimized_approx_equal_slopes(*last_valid_segment, *next_segment);
            // Do not accept "equal" slopes if at the first breakpoint. Here, optimized_approx_equal_slopes will
            // always be true because the value for ->slope of the first bp is infinite, and thus always approx_eq
            if (bps_have_equal_position || (bps_have_equal_slope && last_valid_segment != breakpoints.begin())) {
                // *first will be overwritten in consecutive operations. Copy first to result to ensure that the last
                // seen value of *first is kept.
                if (prev_valid_segment != breakpoints.end()) {
                    *last_valid_segment       = *next_segment;
                    last_valid_segment->slope = compute_slope(*prev_valid_segment, *next_segment);
                }
                continue;
            }

            prev_valid_segment = last_valid_segment;

            if (++last_valid_segment != next_segment) {
                *last_valid_segment = std::move(*next_segment);
            }
        }

        breakpoints.erase(++last_valid_segment, breakpoints.end());

        assert(breakpoints.front().slope == PWLFunction::LB_SLOPE);

        if (breakpoints.size() >= 2) {
            // Special case: Segments a - b - c , a will get merged with b
            // Then the slope of c will not be recomputed as last_valid_segment == a and prev_valid_segment == end.
            // It is easier to simply always recompute the slope of the first non-lb breakpoint than to explicity
            // check for this case in the loop
            breakpoints[1].slope = compute_slope(breakpoints.front(), breakpoints[1]);
            return;
        };

        assert(breakpoints.size() == 1);
        // We may have removed more breakpoints than we were allowed to (if LB == UB)
        breakpoints.push_back(breakpoints.front());
        breakpoints.back().slope = 0.0;
    }

    void optimize_breakpoint_sequence(PWLFunction& function) { optimize_breakpoint_sequence(function._segments); }

    unsigned long PWLFunction::getNumberOfBreakpoints() const { return _segments.size(); }

    PWLFunction create_constant_pwl(double min_domain, double max_domain, double img) {
        return PWLFunction(PWLFunction::bp_container_t { { min_domain, img }, { max_domain, img, 0.0 } });
    }

    PWLFunction create_single_point_pwl(double domain, double img) { return create_constant_pwl(domain, domain, img); }

    PWLFunction construct_from_breakpoints(
        PWLFunction::bp_container_t breakpoints, bool optimize, bool force_recomputation) {

        if (force_recomputation) {
            recompute_slopes(breakpoints);
        }

        if (optimize) {
            optimize_breakpoint_sequence(breakpoints);
        }

        return PWLFunction(std::move(breakpoints));
    }

    bool is_flat(const PWLFunction& function) {
        return approx_eq(function.getImageLowerBound(), function.getImageUpperBound());
    }

    std::size_t hash_value(const PWLFunction::breakpoint_t& bp) {
        std::size_t hash = 0;
        boost::hash_combine(hash, bp.domain);
        boost::hash_combine(hash, bp.image);
        boost::hash_combine(hash, bp.slope);
        return hash;
    }

    std::size_t hash_value(const PWLFunction& f) {
        std::size_t hash = 0;
        for (const auto& bp : f.getBreakpoints()) {
            boost::hash_combine(hash, hash_value(bp));
        }
        return hash;
    }

    const PWLFunction::bp_container_t& get_breakpoints(const PWLFunction& func) { return func.getBreakpoints(); }

    double value(const PWLFunction& func, double x) { return func.value(x); }

    double inverse(const PWLFunction& func, double y) { return func.inverse(y); }
    double get_lower_bound(const PWLFunction& phi) { return phi.getLowerBound(); }
    double get_upper_bound(const PWLFunction& phi) { return phi.getUpperBound(); }
    double get_image_lower_bound(const PWLFunction& phi) { return phi.getImageLowerBound(); }
    double get_image_upper_bound(const PWLFunction& phi) { return phi.getImageUpperBound(); }
}