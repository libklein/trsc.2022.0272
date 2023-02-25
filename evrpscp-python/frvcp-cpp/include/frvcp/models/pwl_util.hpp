
#ifndef FRVCP_PWL_UTIL_HPP
#define FRVCP_PWL_UTIL_HPP

#include "frvcp/util/algorithms.hpp"
#include "pwl.hpp"
#include <algorithm>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <utility>

namespace frvcp::models {

    /**
     * Shifts the passed PWL function by delta_x and delta_y on x and y axes respectively.
     * @param func The function
     * @param delta_x The amount to shift on the x axis.
     * @param delta_y The amount to shift on the y axis
     * @return The shifted PWLFunction
     */
    template<class Func> [[nodiscard]] Func shift_by(const Func& func, double delta_x, double delta_y) {
        typename Func::bp_container_t breakpoints;
        breakpoints.reserve(get_breakpoints(func).size());

        // Shift all breakpoints
        std::transform(func.begin(), func.end(), std::back_inserter(breakpoints), [delta_x, delta_y](const auto& bp) {
            return decltype(bp) { bp.domain + delta_x, bp.image + delta_y, bp.slope };
        });

        return Func(construct_from_breakpoints(std::move(breakpoints), false, false));
    }

    /**
     * Clips a PWL Function to the specified bounds. The function is well-defined only if the
     * function is below image_lower_bound/above image_upper_bound at it's begin and end.
     * @param func The PWL Function
     * @param image_lower_bound The new lower bound on the image
     * @param image_upper_bound The new upper bound on the image
     * @return A newly constructed, PWLFunction constrained to the specified bounds
     */
    template<class Func>
    [[nodiscard]] Func clip_image(const Func& func, double image_lower_bound, double image_upper_bound) {
#ifdef ENABLE_SAFETY_CHECKS
        if (image_lower_bound > image_upper_bound) {
            throw std::runtime_error("Cannot clip image with higher lower than upper bound!");
        }
        if (image_lower_bound > get_image_upper_bound(func)) {
            throw std::runtime_error("Image-clipping would create an empty PWL function!");
        }
        if (image_upper_bound < get_image_lower_bound(func)) {
            throw std::runtime_error("Image-clipping would create an empty PWL function!");
        }
        {
            // Check pre-conditions
            // All bp with image < image_lb appear at end
            if (!std::is_partitioned(func.begin(), func.end(),
                    [image_lower_bound](const auto& bp) { return bp.image < image_lower_bound; })) {
                throw std::runtime_error("Image-clipping precondition does not hold!");
            }
            // All bp with image > image_ub appear at end
            if (!std::is_partitioned(func.begin(), func.end(),
                    [image_upper_bound](const auto& bp) { return !(bp.image > image_upper_bound); })) {
                throw std::runtime_error("Image-clipping precondition does not hold!");
            }
            // A decreasing function is still partitioned. Verify that the function is not higher at beg than at end
            if (get_breakpoints(func).front().image > get_breakpoints(func).back().image) {
                throw std::runtime_error("Image-clipping precondition does not hold!");
            }
        }
#endif

        const auto& original_bps = get_breakpoints(func);

        typename Func::bp_container_t breakpoints;
        breakpoints.reserve(original_bps.size());

        image_lower_bound = std::max(image_lower_bound, get_image_lower_bound(func));
        image_upper_bound = std::min(image_upper_bound, get_image_upper_bound(func));

        if (image_lower_bound == image_upper_bound) {
            return Func(models::create_single_point_pwl(inverse(func, image_lower_bound), image_lower_bound));
        }

        // Skip any breakpoint <= lb
        auto next_valid_seg = original_bps.begin();
        for (; next_valid_seg->image <= image_lower_bound; ++next_valid_seg) { };

        // next valid segment points to the first segment that will be part of the new function
        // Push the new lb segment
        breakpoints.emplace_back(inverse(func, image_lower_bound), image_lower_bound, models::PWLFunction::LB_SLOPE);

        // Copy all segments that will be part of the new function
        for (; next_valid_seg->image < image_upper_bound; ++next_valid_seg) {
            breakpoints.push_back(*next_valid_seg);
        }

        // Add the ub. next_valid_bp will always point to a valid breakpoint since image_upper_bound <= function_image_ub
        breakpoints.emplace_back(inverse(func, image_upper_bound), image_upper_bound, next_valid_seg->slope);

        assert(breakpoints.size() > 1);

        return Func(construct_from_breakpoints(breakpoints, false, false));
    }

    /**
     * Clips a PWL Function to the specified bounds.
     * @param func The PWL Function
     * @param lower_bound The new lower bound on the domain
     * @param upper_bound The new upper bound on the domain
     * @return A newly constructed, PWLFunction constrained to the specified bounds
     */
    template<class Func> [[nodiscard]] Func clip_domain(const Func& func, double lower_bound, double upper_bound) {
        if (get_lower_bound(func) > upper_bound || get_upper_bound(func) < lower_bound || lower_bound > upper_bound) {
            throw std::runtime_error("Cannot create an empty PWL function.");
        }

        lower_bound = std::max(lower_bound, get_lower_bound(func));
        upper_bound = std::min(upper_bound, get_upper_bound(func));

        if (lower_bound == upper_bound) {
            return Func(models::create_single_point_pwl(lower_bound, value(func, lower_bound)));
        }

        const auto& original_breakpoints = func.getBreakpoints();

        typename Func::bp_container_t breakpoints;
        breakpoints.reserve(original_breakpoints.size());

        auto cut_segment = [&func](double domain, double slope) {
            return typename Func::breakpoint_t { domain, value(func, domain), slope };
        };

        // Scan over function, adding any breakpoint that is within lower/upper bound
        auto next_valid_bp = original_breakpoints.begin();

        // Skip any BP where the domain is <= lower_bound
        for (; next_valid_bp->domain <= lower_bound; ++next_valid_bp) { }

        // Add new LB
        breakpoints.emplace_back(lower_bound, value(func, lower_bound), models::PWLFunction::LB_SLOPE);

        // Add breakpoints until next_valid_bp->domain >= upper_bound
        for (; next_valid_bp != original_breakpoints.end() && next_valid_bp->domain < upper_bound; ++next_valid_bp) {
            breakpoints.push_back(*next_valid_bp);
        }

        // Insert the new UB
        breakpoints.push_back(cut_segment(upper_bound, next_valid_bp->slope));

        assert(breakpoints.size() > 1);

        return Func(construct_from_breakpoints(std::move(breakpoints), false, false));
    }

    template<class Breakpoint> [[nodiscard]] double compute_slope(const Breakpoint& pred, const Breakpoint& cur) {
        if (cur.domain == pred.domain) {
            return 0.0;
        }
        double slope = (cur.image - pred.image) / (cur.domain - pred.domain);
        if (std::abs(slope) > frvcp::MAX_SLOPE || std::abs(slope) < frvcp::MIN_SLOPE) {
            return 0.0;
        }
        return slope;
    }

    /***
     * Removes redundant breakpoints from a piecewise linear function. Specifically, this removes segements smaller than the
     * minimum precision and joins consecutive segments with the same slope.
     * @param func The PWL function.
     * @return An optimized copy of the function.
     */
    template<class Func> [[nodiscard]] Func remove_redundant_breakpoints(const Func& func) {
        // typename Func::bp_container_t breakpoints = func.getBreakpoints();
        const auto& original_breakpoints = func.getBreakpoints();
        assert(original_breakpoints.size() >= 2);
        typename Func::bp_container_t breakpoints;
        breakpoints.reserve(original_breakpoints.size());

        auto next_bp = original_breakpoints.begin();
        breakpoints.push_back(*next_bp);
        while (++next_bp != original_breakpoints.end()) {
            const auto& prev_bp = breakpoints.back();

            if (approx_eq(prev_bp.domain, next_bp->domain) && approx_eq(prev_bp.image, next_bp->image)) {
                continue;
            }

            assert(breakpoints.capacity() > breakpoints.size());
            breakpoints.push_back(*next_bp);
            breakpoints.back().slope = compute_slope(prev_bp, *next_bp);
        }

        auto new_end = unique_keep_last(breakpoints.begin(), breakpoints.end(),
            [](const auto& prev, const auto& next) { return prev.slope == next.slope; });
        breakpoints.erase(new_end, breakpoints.end());

        return Func(std::move(breakpoints));
    }
}

#endif // EVSPNL_SUBPROBLEM_PWL_UTIL_HPP
