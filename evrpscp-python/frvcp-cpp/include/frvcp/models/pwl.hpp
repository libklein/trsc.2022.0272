
#ifndef FRVCP_PWL_HPP
#define FRVCP_PWL_HPP

#include "frvcp/definitions.hpp"
#include "frvcp/util/floats.hpp"
#include <algorithm>
#include <compare>
#include <ostream>
#include <ranges>
#include <vector>

#include <boost/container/small_vector.hpp>

namespace frvcp::models {

    void enable_throw_on_extensive_checks(bool enable);

    struct Segment {
        double domain; // Upper bound on domain axis
        double image; // Upper bound on image axis
        double slope; // Slope of segment between previous and this BP. Infinity if no previous BP

        Segment(double domain, double image);
        Segment(double domain, double image, double slope);

        [[nodiscard]] std::strong_ordering operator<=>(const Segment& other) const;
        [[nodiscard]] bool operator==(const Segment& other) const = default;

        friend std::ostream& operator<<(std::ostream& os, const Segment& breakpoint);
    };

    class PWLFunction {
    public:
        constexpr static double LB_SLOPE = std::numeric_limits<double>::infinity();
        //__attribute__((aligned(32)));
    public:
        using breakpoint_t   = Segment;
        using bp_container_t = boost::container::small_vector<Segment, 10>;
        using iterator       = bp_container_t::iterator;
        using const_iterator = bp_container_t::const_iterator;

    private:
        bp_container_t _segments;
        double _minimum_image;
        double _maximum_image;

    protected:
        [[nodiscard]] bp_container_t& _get_segments();
        [[nodiscard]] const bp_container_t& _get_segments() const;

        [[nodiscard]] bp_container_t::const_iterator _find_segment(double domain_value) const;
        [[nodiscard]] bp_container_t::const_iterator _find_inverse_segment(double image_value) const;

        PWLFunction() = default;

    public:
        explicit PWLFunction(bp_container_t breakpoints);

        [[nodiscard]] const bp_container_t& getBreakpoints() const;
        [[nodiscard]] bp_container_t::size_type getNumberOfBreakpoints() const;

        [[nodiscard]] double getImageUpperBound() const;
        [[nodiscard]] double getUpperBound() const;

        [[nodiscard]] double getImageLowerBound() const;
        [[nodiscard]] double getLowerBound() const;

        [[nodiscard]] iterator begin();
        [[nodiscard]] const_iterator begin() const;

        [[nodiscard]] iterator end();
        [[nodiscard]] const_iterator end() const;

        std::strong_ordering operator<=>(const PWLFunction& rhs) const;
        bool operator==(const PWLFunction& rhs) const;

        friend std::ostream& operator<<(std::ostream& os, const PWLFunction& function);

        /// Return f(x)
        [[nodiscard]] double operator()(double x) const;
        /// Return f(x)
        [[nodiscard]] double value(double x) const;

        /// Returns f^-1(x)
        [[nodiscard]] double inverse(double x) const;

        /// Returns the slope at x
        [[nodiscard]] double slope(double x) const;

        /// Returns the slope at y
        [[nodiscard]] double slope_at_inverse(double y) const;

        friend void optimize_breakpoint_sequence(PWLFunction& function);
    };

    [[nodiscard]] std::size_t hash_value(const PWLFunction::breakpoint_t& bp);
    [[nodiscard]] std::size_t hash_value(const PWLFunction& f);

    /**
     * Creates a PWL function with value f(x = domain) = img, and undefined for any other x.
     * @param domain Value on x axis
     * @param img Value on y axis
     * @return Single point PWL Function
     */
    [[nodiscard]] PWLFunction create_single_point_pwl(double domain, double img);

    /**
     * Creates a constant (i.e., flat) PWL function.
     * @param min_domain First value where the function is defined.
     * @param max_domain Last value where the function is defined.
     * @param img Value on the y axis, i.e. f(x) = y \forall x \in [min_domain, max_domain]
     * @return
     */
    [[nodiscard]] PWLFunction create_constant_pwl(double min_domain, double max_domain, double img);

    template<class Func> bool is_concave(const Func& func) {
        return (std::adjacent_find(std::next(func.begin()), func.end(), [](const auto& lhs_bp, const auto& rhs_bp) {
            return certainly_lt(lhs_bp.slope, rhs_bp.slope);
        })) == func.end();
    }

    template<class Func> bool is_convex(const Func& func) {
        return (std::adjacent_find(std::next(func.begin()), func.end(), [](const auto& lhs_bp, const auto& rhs_bp) {
            return certainly_gt(lhs_bp.slope, rhs_bp.slope);
        })) == func.end();
    }

    template<class Func> bool is_increasing(const Func& f) {
        return (std::adjacent_find(std::next(f.begin()), f.end(), [](const auto& lhs_bp, const auto& rhs_bp) {
            return certainly_gt(lhs_bp.image, rhs_bp.image);
        })) == f.end();
    }

    bool is_flat(const PWLFunction& function);

    void optimize_breakpoint_sequence(PWLFunction::bp_container_t& segments);

    /**
     * Optimize
     * @param function
     */
    void optimize_breakpoint_sequence(PWLFunction& function);

    /**
     * Construct a PWL function from a set of breakpoints.
     * @param breakpoints
     * @param optimize
     * @param force_recomputation
     * @return The newly constructed PWL function
     */
    [[nodiscard]] PWLFunction construct_from_breakpoints(
        PWLFunction::bp_container_t breakpoints, bool optimize = true, bool force_recomputation = false);

    void recompute_slopes(frvcp::models::PWLFunction::bp_container_t& breakpoints);

    [[nodiscard]] const PWLFunction::bp_container_t& get_breakpoints(const PWLFunction& func);
    [[nodiscard]] double value(const PWLFunction& func, double x);
    [[nodiscard]] double inverse(const PWLFunction& func, double y);
    [[nodiscard]] double get_lower_bound(const PWLFunction& phi);
    [[nodiscard]] double get_upper_bound(const PWLFunction& phi);
    [[nodiscard]] double get_image_lower_bound(const PWLFunction& phi);
    [[nodiscard]] double get_image_upper_bound(const PWLFunction& phi);
}

#endif // FRVCP_PWL_HPP
