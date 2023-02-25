
#ifndef FRVCP_COST_PROFILE_HPP
#define FRVCP_COST_PROFILE_HPP

#include "models/fwd.hpp"
#include "models/pwl.hpp"
#include <optional>
#include <ostream>

namespace frvcp {

    class CostProfile {
        /***
         * Captures Cost-SoC mapping when charging at some station.
         * Assume concave cost profiles
         */
    public:
        // using bp_container_t = std::vector<models::PWLFunction::breakpoint_t>;
        using bp_container_t = models::PWLFunction::bp_container_t;
        using iterator       = bp_container_t::iterator;
        using const_iterator = bp_container_t::const_iterator;

    private:
        models::PWLFunction _cost_soc_mapping;

    public:
        CostProfile();
        explicit CostProfile(bp_container_t breakpoints);
        explicit CostProfile(models::PWLFunction func);

        /// Minimum cost required such that path traversal is feasible.
        [[nodiscard]] double getMinimumCost() const;
        /// Least cost to reach the end of the path with maximal soc
        [[nodiscard]] double getMaximumCost() const;
        /// Minimum SoC at end of path
        [[nodiscard]] double getMinimumSoC() const;
        /// Maximum SoC at end of path
        [[nodiscard]] double getMaximumSoC() const;

        [[nodiscard]] iterator begin();
        [[nodiscard]] const_iterator begin() const;

        [[nodiscard]] iterator end();
        [[nodiscard]] const_iterator end() const;

        [[nodiscard]] const bp_container_t& getBreakpoints() const;

        bool operator==(const CostProfile& rhs) const;
        bool operator!=(const CostProfile& rhs) const;

        [[nodiscard]] double value(double cost) const;
        [[nodiscard]] double inverse(double soc) const;
        [[nodiscard]] double operator()(double cost) const;

        friend std::ostream& operator<<(std::ostream& os, const CostProfile& profile);
        friend void optimize_breakpoint_sequence(CostProfile& profile);
    };

    [[nodiscard]] const CostProfile::bp_container_t& get_breakpoints(const CostProfile& func);
    [[nodiscard]] double value(const CostProfile& func, double x);
    [[nodiscard]] double inverse(const CostProfile& func, double y);
    [[nodiscard]] double get_lower_bound(const CostProfile& phi);
    [[nodiscard]] double get_upper_bound(const CostProfile& phi);
    [[nodiscard]] double get_image_lower_bound(const CostProfile& phi);
    [[nodiscard]] double get_image_upper_bound(const CostProfile& phi);
    [[nodiscard]] double getSoCAtCost(const CostProfile& profile, double cost);
    [[nodiscard]] double getCostAtSoC(const CostProfile& profile, double soc);

    /**
     * Optimizes the given cost profile based on the underlying cost-soc mapping.
     * @param profile The cost profile to be optimized
     */
    void optimize_breakpoint_sequence(CostProfile& profile);

    /**
     * Creates a flat Cost->SoC mapping, i.e., a mapping which gives initial_soc for all c >= initial_cost, and -\infty
     * otherwise.
     * @param initial_cost Sunk cost.
     * @param initial_soc Only reachable state of charge.
     * @return A flat Cost-SoC mapping.
     */
    CostProfile create_flat_profile(double initial_cost, double initial_soc);

    /**
     * Creates a new cost profile by shifting profile according to delta_cost and delta_soc, such that no infeasible
     * segments remain. Essentially computes profile' = profile + (delta_cost, delta_soc)
     * Precondition:
     *  Profile shifted by (delta_cost, delta_soc) has at least one feasible point.
     * @param profile The cost profile
     * @param delta_cost The amount to shift on the x (cost) axis.
     * @param delta_soc The amount to shift on the y (soc) axis
     * @param upper_cutoff_soc Cut-off value for soc. Anything above this will be discarded. Should be set to max_soc in most
     * cases
     * @return The shifted cost profile
     */
    CostProfile shift_by(const CostProfile& profile, double delta_cost, double delta_soc, double upper_cutoff_soc);

    /**
     * Computes the Cost->SoC mapping when charging with an initially empty battery according to a given wear cost density
     * function wdf and energy price energy_price.
     * @param wdf The wear cost density function.
     * @param energy_price The energy price.
     * @return A Cost->SoC mapping according to wdf and energy_price
     */
    CostProfile create_period_profile(const models::WearCostDensityFunction& wdf, double energy_price);

    /**
     * Computes the cost->soc mapping when charging according to station_cost_profile with sunk costs entry_cost
     * and soc entry_soc and a maximum charge amount of max_delta_soc.
     * @param station_cost_profile The Cost->SoC mapping at the station when entering with an empty battery and no sunk cost.
     * @param max_delta_soc The maximum amount that can be recharged.
     * @param entry_cost The sunk cost upon starting to charge.
     * @param entry_soc The initial soc to start charging with.
     * @return A cost profile capturing the charging decisions according to station_cost_profile.
     */
    CostProfile replace_station(
        const CostProfile& station_cost_profile, double max_delta_soc, double entry_cost, double entry_soc);

    /**
     * Derives a new cost profile which captures all cost<->soc trade-offs when charging at an intermediate station i.
     * @param prev_exit_soc Cost profile of previous, i.e., the entry SoC at i.
     * @param station_cost_profile The cost profile of i.
     * @param phi The charging function of i.
     * @param fix_cost Fix cost of charging at i.
     * @param committed_charge_time The time to charge at i.
     * @param max_soc The maximum SoC up to which to charge.
     * @return The new, optimized, cost profile.
     */
    frvcp::CostProfile charge_at_intermediate_station(const CostProfile& prev_exit_soc,
        const CostProfile& station_cost_profile, const frvcp::models::ChargingFunction& phi, double fix_cost,
        double committed_charge_time, double max_soc);

    frvcp::CostProfile simulate_charge_at_intermediate_station(const CostProfile& prev_exit_soc,
        const CostProfile& station_cost_profile, double fix_cost, const frvcp::models::ChargingFunction& phi,
        double max_soc);

    double cost_of_intermediate_charge(const CostProfile& prev_exit_soc, const CostProfile& station_cost_profile,
        double fix_cost, const frvcp::models::ChargingFunction& phi, double charge_time, double target_soc);

    std::vector<frvcp::CostProfile> charge_intermediatly(const CostProfile& profile, const CostProfile& station_cost_profile,
        const models::ChargingFunction& phi, double fixed_cost, double duration, double max_soc);

    bool is_flat(const CostProfile& profile);
}
#endif // FRVCP_COST_PROFILE_HPP
