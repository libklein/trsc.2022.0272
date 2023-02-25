#include "frvcp/cost_profile.hpp"
#include "frvcp/models/battery.hpp"
#include "frvcp/models/charger.hpp"
#include "frvcp/models/pwl.hpp"
#include "frvcp/models/pwl_util.hpp"
#include "frvcp/util/algorithms.hpp"
#include "frvcp/util/floats.hpp"

#include <algorithm>
#include <cassert>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <iostream>
#include <vector>

using namespace frvcp;
using namespace frvcp::models;

namespace {
    CostProfile construct_cost_profile_from_breakpoints(
        CostProfile::bp_container_t breakpoints, bool optimize = true, bool force_recomputation = false) {
        return CostProfile(construct_from_breakpoints(std::move(breakpoints), optimize, force_recomputation));
    }

    std::vector<CostProfile> create_cost_profiles_from_bp_vectors(
        std::vector<std::vector<PWLFunction::breakpoint_t>> breakpoints, double max_soc, bool optimize = false,
        bool recompute_slopes = false) {
        std::vector<CostProfile> profiles;
        profiles.reserve(breakpoints.size());

        for (auto& bp_container : breakpoints) {
            bp_container.front().slope = models::PWLFunction::LB_SLOPE;
            // TODO Clip here is inefficient!
            profiles.push_back(clip_image(
                construct_cost_profile_from_breakpoints(
                    PWLFunction::bp_container_t { bp_container.begin(), bp_container.end() }, optimize, recompute_slopes),
                0.0, max_soc));
        }

        return profiles;
    }

    void recompute_slopes(frvcp::models::PWLFunction::bp_container_t& breakpoints) {
        if (breakpoints.size() < 2)
            return;
        auto prev = breakpoints.begin();
        for (auto cur = std::next(prev); cur != breakpoints.end(); prev = cur++) {
            cur->slope = (cur->image - prev->image) / (cur->domain - prev->domain);
        }
    }
}

namespace frvcp {

    [[nodiscard]] double getSoCAtCost(const CostProfile& profile, double cost) { return profile.value(cost); }

    [[nodiscard]] double getCostAtSoC(const CostProfile& profile, double soc) { return profile.inverse(soc); }

    std::ostream& operator<<(std::ostream& os, const CostProfile& profile) {
        os << "Cost profile: \n" << profile._cost_soc_mapping;
        return os;
    }

    frvcp::CostProfile::CostProfile()
        : CostProfile(models::create_single_point_pwl(COST_OF_UNREACHABLE_SOC, 0.0)) { }

    frvcp::CostProfile::CostProfile(frvcp::CostProfile::bp_container_t breakpoints)
        : CostProfile(PWLFunction(std::move(breakpoints))) { }

    CostProfile::CostProfile(models::PWLFunction func)
        : _cost_soc_mapping(std::move(func)) { }

    double frvcp::CostProfile::getMinimumCost() const { return _cost_soc_mapping.getLowerBound(); }

    double frvcp::CostProfile::getMaximumCost() const { return _cost_soc_mapping.getUpperBound(); }

    double frvcp::CostProfile::getMinimumSoC() const { return _cost_soc_mapping.getImageLowerBound(); }

    double frvcp::CostProfile::getMaximumSoC() const { return _cost_soc_mapping.getImageUpperBound(); }

    const frvcp::CostProfile::bp_container_t& frvcp::CostProfile::getBreakpoints() const {
        return _cost_soc_mapping.getBreakpoints();
    }

    bool frvcp::CostProfile::operator==(const frvcp::CostProfile& rhs) const {
        return getBreakpoints() == rhs.getBreakpoints();
    }

    bool frvcp::CostProfile::operator!=(const frvcp::CostProfile& rhs) const {
        return getBreakpoints() != rhs.getBreakpoints();
    }

    double frvcp::CostProfile::value(double cost) const {
        if (cost < getMinimumCost()) {
            return INVALID_SOC;
        } else if (cost >= getMaximumCost()) {
            return getMaximumSoC();
        } else {
            return _cost_soc_mapping.value(cost);
        }
    }

    double frvcp::CostProfile::inverse(double soc) const {
        if (soc <= getMinimumSoC()) {
            return getMinimumCost();
        } else if (soc > getMaximumSoC()) {
            return COST_OF_UNREACHABLE_SOC;
        } else {
            return _cost_soc_mapping.inverse(soc);
        }
    }

    double frvcp::CostProfile::operator()(double cost) const { return value(cost); }

    CostProfile::iterator CostProfile::begin() { return _cost_soc_mapping.begin(); }

    CostProfile::const_iterator CostProfile::begin() const { return _cost_soc_mapping.begin(); }

    CostProfile::iterator CostProfile::end() { return _cost_soc_mapping.end(); }

    CostProfile::const_iterator CostProfile::end() const { return _cost_soc_mapping.end(); }

    frvcp::CostProfile create_flat_profile(double initial_cost, double initial_soc) {
        // Create a single point PWL. Values c != initial_cost will be handled by the profile.
        return frvcp::CostProfile(create_single_point_pwl(initial_cost, initial_soc));
    }

    void optimize_breakpoint_sequence(CostProfile& cost_profile) {
        models::optimize_breakpoint_sequence(cost_profile._cost_soc_mapping);
    }

    CostProfile create_period_profile(const WearCostDensityFunction& wdf, double energy_price) {
        assert(wdf.getMinimumSoC() == 0.0 && wdf.getMinimumCost() == 0.0);
        assert(wdf.getBreakpoints()[0].domain == 0.0 && wdf.getBreakpoints()[0].image == 0.0);
        frvcp::CostProfile::bp_container_t breakpoints;
        breakpoints.emplace_back(0.0, 0.0);

        double prev_soc  = 0.0;
        double prev_cost = 0.0;
        for (auto bp_iter = std::next(wdf.begin()); bp_iter != wdf.end(); ++bp_iter) {
            double soc = bp_iter->domain;
            // Cost profile maps cost -> soc, hence slope has unit soc/cost
            double slope = 1.0 / (bp_iter->slope + energy_price);
            double cost  = prev_cost + (soc - prev_soc) / slope;
            breakpoints.emplace_back(cost, soc, slope);
            prev_cost = cost, prev_soc = soc;
        }

        // Avoid numerical instability. This requires slope recomputation.
        breakpoints.back().image = wdf.getMaximumSoC();

        // std::sort(breakpoints.begin(), breakpoints.end());

        return construct_cost_profile_from_breakpoints(std::move(breakpoints), true, true);
    }

    frvcp::CostProfile replace_station(
        const CostProfile& station_cost_profile, double max_delta_soc, double entry_cost, double entry_soc) {
        /*
         * (entry_cost, entry_soc) will be our new "origin", i.e., we shift by (entry_cost -
         * station_cost_profile^-1(entry_soc), 0), cut off any breakpoints which charge more than can be charged in the time
         * available, and finally cut off any soc < entry_soc
         */
        double cost_offset   = entry_cost - station_cost_profile.inverse(entry_soc);
        auto shifted_profile = shift_by(
            station_cost_profile, cost_offset, 0, std::min(entry_soc + max_delta_soc, station_cost_profile.getMaximumSoC()));

        frvcp::CostProfile::bp_container_t bps;
        // New lower bound, i.e., all values below entry_soc are cut off
        bps.emplace_back(entry_cost, entry_soc);
        std::copy_if(shifted_profile.begin(), shifted_profile.end(), std::back_inserter(bps),
            [entry_soc](const auto& bp) { return bp.image > entry_soc; });

        // Special case - only a single breakpoint. Happens when entry_soc == min(station_cost_profile.getMaximumSoC(),
        // entry_soc + max_delta_soc)
        if (bps.size() == 1) {
            bps.emplace_back(bps.back().domain, bps.back().image, 0.0);
        }

        return construct_cost_profile_from_breakpoints(std::move(bps), true);
    }

    double cost_of_intermediate_charge(const CostProfile& prev_exit_soc, const CostProfile& station_cost_profile,
        double fix_cost, const ChargingFunction& phi, double charge_time, double target_soc) {
        double entry_soc = initial_soc_when_charging(phi, target_soc, charge_time);
        return prev_exit_soc.inverse(entry_soc) + fix_cost
            + (station_cost_profile.inverse(target_soc) - station_cost_profile.inverse(entry_soc));
    }

    frvcp::CostProfile simulate_charge_at_intermediate_station(const CostProfile& prev_exit_soc,
        const CostProfile& station_cost_profile, double fix_cost, const ChargingFunction& phi, double max_soc) {
        double STEPSIZE = 0.001;
        CostProfile::bp_container_t breakpoints;
        breakpoints.reserve(
            static_cast<std::size_t>((prev_exit_soc.getMaximumCost() - prev_exit_soc.getMinimumCost()) / STEPSIZE));

        auto calculate_breakpoint = [&](double cost_at_f) {
            double exit_soc = phi.getCharge(prev_exit_soc(cost_at_f), PERIOD_LENGTH);
            exit_soc        = std::min(max_soc, exit_soc);

            double cost
                = cost_of_intermediate_charge(prev_exit_soc, station_cost_profile, fix_cost, phi, PERIOD_LENGTH, exit_soc);
            return CostProfile::bp_container_t::value_type(cost, exit_soc);
        };

        double prev_simulated_exit_soc = std::numeric_limits<double>::lowest();
        double next_cost_at_f          = prev_exit_soc.getMinimumCost();
        while (next_cost_at_f <= prev_exit_soc.getMaximumCost()) {
            auto new_bp = calculate_breakpoint(next_cost_at_f);
            if (!approx_eq(prev_simulated_exit_soc, new_bp.image)) {
                breakpoints.push_back(new_bp);
                prev_simulated_exit_soc = new_bp.image;
            }
            if (new_bp.image >= max_soc)
                break;

            next_cost_at_f += STEPSIZE;
        }

        // Final breakpoint
        breakpoints.push_back(calculate_breakpoint(next_cost_at_f));

        return construct_cost_profile_from_breakpoints(std::move(breakpoints), true, true);
    }

    std::vector<frvcp::CostProfile> charge_intermediatly(const CostProfile& profile, const CostProfile& station_cost_profile,
        const models::ChargingFunction& phi, double fixed_cost, double duration, double max_soc) {
        // Force charging for duration at a station at cost station_cost_profile and charging function phi.
        // We will implicitly compute the resulting cost profile based on the breakpoints of the inverse's derivative.
        static std::vector<double> delta_soc_on_label;
        delta_soc_on_label.clear();

        const auto cost_of_charging_delta_at_prev
            = [&profile, &phi, &station_cost_profile, &duration, &fixed_cost](auto delta_soc) {
                  assert(approx_lt(profile.getMinimumSoC() + delta_soc, profile.getMaximumSoC()));
                  // Create breakpoint from charge amount at incoming label
                  // Avoid rounding errors
                  const double entry_soc = std::min(profile.getMinimumSoC() + delta_soc, profile.getMaximumSoC());
                  const double exit_soc
                      = std::min(charge_for_time(phi, entry_soc, duration), station_cost_profile.getMaximumSoC());

                  const double delta_cost_at_intermediate_station
                      = getCostAtSoC(station_cost_profile, exit_soc) - getCostAtSoC(station_cost_profile, entry_soc);

                  const double entry_cost = getCostAtSoC(profile, entry_soc);
                  const double exit_cost  = entry_cost + delta_cost_at_intermediate_station + fixed_cost;
                  assert(exit_cost != COST_OF_UNREACHABLE_SOC);
                  assert(std::isfinite(exit_cost));
                  assert(std::isfinite(exit_soc));
                  return CostProfile::bp_container_t::value_type { exit_cost, exit_soc };
              };

        const auto exit_soc_with_entry_soc
            = [&phi, duration](auto entry_soc) { return charge_for_time(phi, entry_soc, duration); };
        const auto entry_soc_with_exit_soc
            = [&phi, duration](auto exit_soc) { return initial_soc_when_charging(phi, exit_soc, duration); };

        double const min_entry_soc = profile.getMinimumSoC();
        double const min_exit_soc  = charge_for_time(phi, min_entry_soc, duration);
        double const max_exit_soc
            = std::min(station_cost_profile.getMaximumSoC(), charge_for_time(phi, profile.getMaximumSoC(), duration));
        // Ensure that we charge as much as possible at the second station.
        double const max_entry_soc = entry_soc_with_exit_soc(max_exit_soc);

        // Two reasons for breakpoints of the station_cost_profile causing breakpoints on the final profile:
        //  * Exit SoC passes breakpoint
        //  * Entry SoC passes breakpoint
        for (const auto& station_bp : station_cost_profile) {
            auto bp_soc = station_bp.image;
            if (min_exit_soc <= bp_soc && bp_soc <= max_exit_soc) {
                // ^ Does the exit SoC ever pass this breakpoint?
                // Normalize
                delta_soc_on_label.push_back(entry_soc_with_exit_soc(bp_soc) - min_entry_soc);
            } else if (min_entry_soc <= bp_soc && bp_soc <= max_entry_soc) {
                // ^ Does the entry SoC ever pass this breakpoint?
                // Normalize
                delta_soc_on_label.push_back(bp_soc - min_entry_soc);
            }
        }

        // Charging rate at intermediate charging station changes. This changes the amount of charge that can be replished
        // within duration.
        //  * Entry SoC
        //  * Exit SoC
        for (const auto& phi_bp : phi) {
            auto bp_soc = phi_bp.image;
            if (min_exit_soc <= bp_soc && bp_soc <= max_exit_soc) {
                // ^ Does the exit SoC ever pass this breakpoint?
                // Transform to delta soc
                delta_soc_on_label.push_back(entry_soc_with_exit_soc(bp_soc) - min_entry_soc);
            } else if (min_entry_soc <= bp_soc && bp_soc <= max_entry_soc) {
                // ^ Does the entry SoC ever pass this breakpoint?
                // Transform to delta soc
                delta_soc_on_label.push_back(bp_soc - min_entry_soc);
            }
        }

        // Finally, points where the derivative of the entry cost profile changes may be breakpoints as well
        for (const auto& label_bp : profile.getBreakpoints()) {
            // Breakpoints give the SoC at the end of the path associated with label, which is our entry SoC.
            double const entry_soc_bp = label_bp.image;
            if (entry_soc_bp >= max_entry_soc) {
                continue;
            }
            // No need to check against min/max entry soc. Those correspond to the cost profile's boundaries anyway.
            delta_soc_on_label.push_back(entry_soc_bp - min_entry_soc);
        }

        // Max entry soc is a breakpoint as well but may not be captured by other breakpoints
        delta_soc_on_label.push_back(max_entry_soc - min_entry_soc);

        for (double const delta_soc : delta_soc_on_label) {
            assert(delta_soc >= 0.0);
        }

        std::sort(delta_soc_on_label.begin(), delta_soc_on_label.end());
        auto unique_delta_soc_end = std::unique(delta_soc_on_label.begin(), delta_soc_on_label.end());

        CostProfile::bp_container_t breakpoints_of_new_profile;
        breakpoints_of_new_profile.reserve(
            static_cast<size_t>(std::distance(delta_soc_on_label.begin(), unique_delta_soc_end)));

        std::transform(delta_soc_on_label.begin(), unique_delta_soc_end, std::back_inserter(breakpoints_of_new_profile),
            cost_of_charging_delta_at_prev);

        assert(std::is_sorted(breakpoints_of_new_profile.begin(), breakpoints_of_new_profile.end(),
            [](const PWLFunction::breakpoint_t& lhs, const PWLFunction::breakpoint_t& rhs) {
                return lhs.image < rhs.image;
            }));

        // Sort by SoC - we know that a cost profile's SoC is non-decreasing.
        std::sort(breakpoints_of_new_profile.begin(), breakpoints_of_new_profile.end(),
            [](const PWLFunction::breakpoint_t& lhs, const PWLFunction::breakpoint_t& rhs) {
                return lhs.image < rhs.image;
            });

        // Remove consecutive breakpoints with the same domain
        auto unique_bp_end = util::unique_keep_last(breakpoints_of_new_profile.begin(), breakpoints_of_new_profile.end(),
            [](const PWLFunction::breakpoint_t& lhs, const PWLFunction::breakpoint_t& rhs) {
                return approx_eq(lhs.domain, rhs.domain);
            });
        breakpoints_of_new_profile.erase(unique_bp_end, breakpoints_of_new_profile.end());

        // Remove any segments with negative slope. This is correct since negative slopes imply that charging using the
        // input profile is cheaper.
        auto first_positive_slope = std::adjacent_find(breakpoints_of_new_profile.begin(), breakpoints_of_new_profile.end(),
            [](const PWLFunction::breakpoint_t& prev, const PWLFunction::breakpoint_t& next) {
                return prev.domain < next.domain;
            });
        // We can abort if no segments with positive slope exist -> The incoming (or replaced) profile will always dominate
        // this one.
        if (first_positive_slope == breakpoints_of_new_profile.end()) {
            return {};
        }
        breakpoints_of_new_profile.erase(breakpoints_of_new_profile.begin(), first_positive_slope);

        // Remove any tailing segments with negative slope
        auto last_positive_slope = std::adjacent_find(breakpoints_of_new_profile.rbegin(), breakpoints_of_new_profile.rend(),
            [](const PWLFunction::breakpoint_t& next, const PWLFunction::breakpoint_t& prev) {
                return next.domain > prev.domain;
            });
        // We want to keep the last_positive_slope segment, i.e., remove any segment behind that one.
        // rev_iter.base() always gives the element behind the one rev_iter points at, i.e., a rev iter pointing to the
        // 2nd element would yield an iterator to the 3rd element.
        breakpoints_of_new_profile.erase(last_positive_slope.base(), breakpoints_of_new_profile.end());

        models::recompute_slopes(breakpoints_of_new_profile);

        auto breakpoints_of_new_profile_backup = breakpoints_of_new_profile;
        auto split = util::split_if(breakpoints_of_new_profile.begin(), breakpoints_of_new_profile.end(),
            [](const auto& bp) { return bp.slope <= 0.0; });
        // Remove split vectors where all bps are negative or have a zero slope. We need to account for LB_SLOPE here as
        // well: Consider the case where we have segments [0, 0, inf], [1, -1, -1], [2, 0, 1]. Then the result of calling
        // split will be {[0, 0, inf]}, {[1, -1, -1], [2, 0, 1]}. The first vector does not form a valid PWL and thus
        // needs to be discarded before calling create_pwls_from_split
        split.erase(std::remove_if(split.begin(), split.end(),
                        [](const auto& vec) {
                            return std::all_of(vec.begin(), vec.end(),
                                [](const auto& bp) { return bp.slope <= 0.0 || bp.slope == models::PWLFunction::LB_SLOPE; });
                        }),
            split.end());

        return create_cost_profiles_from_bp_vectors(std::move(split), max_soc, false, false);
    }

    frvcp::CostProfile charge_at_intermediate_station(const CostProfile& prev_exit_soc,
        const CostProfile& station_cost_profile, const ChargingFunction& phi, double fix_cost, double committed_charge_time,
        double max_soc) {
        double min_entry_soc = prev_exit_soc.getMinimumSoC();
        double min_exit_soc  = phi.getCharge(min_entry_soc, committed_charge_time);
        double max_entry_soc
            = std::min(phi.getSoCAfter(phi.getTimeRequired(max_soc) - committed_charge_time), prev_exit_soc.getMaximumSoC());
        double max_exit_soc = std::min(max_soc, phi.getCharge(max_entry_soc, committed_charge_time));

        frvcp::CostProfile::bp_container_t breakpoints;

        // Add the breakpoints of prev profile
        for (const auto& bp_of_prev_profile : prev_exit_soc) {
            double bp_entry_soc = bp_of_prev_profile.image, bp_entry_cost = bp_of_prev_profile.domain;
            if (bp_entry_soc >= max_entry_soc)
                break;
            double bp_exit_soc = charge_for_time(phi, bp_entry_soc, committed_charge_time);
            breakpoints.emplace_back(bp_entry_cost + fix_cost
                    + (station_cost_profile.inverse(bp_exit_soc) - station_cost_profile.inverse(bp_entry_soc)),
                bp_exit_soc);
        }

        // Add the breakpoints of the station cost profile (exit soc)
        for (const auto& bp_of_station_profile : station_cost_profile) {
            double exit_soc = bp_of_station_profile.image, station_profile_exit_cost = bp_of_station_profile.domain;
            if (exit_soc <= min_exit_soc)
                continue;
            if (exit_soc >= max_exit_soc)
                break;
            double entry_soc = initial_soc_when_charging(phi, exit_soc, committed_charge_time);
            double exit_cost = prev_exit_soc.inverse(entry_soc) + fix_cost
                + (station_profile_exit_cost - station_cost_profile.inverse(entry_soc));
            breakpoints.emplace_back(exit_cost, exit_soc);
        }

        // Add the breakpoints of the station cost profile (entry soc)
        for (const auto& bp_of_station_profile : station_cost_profile) {
            double bp_entry_soc = bp_of_station_profile.image, bp_cost = bp_of_station_profile.domain;
            if (bp_entry_soc <= min_entry_soc)
                continue;
            if (bp_entry_soc >= max_entry_soc)
                break;
            double bp_exit_soc = charge_for_time(phi, bp_entry_soc, committed_charge_time);
            double cost
                = prev_exit_soc.inverse(bp_entry_soc) + fix_cost + (station_cost_profile.inverse(bp_exit_soc) - bp_cost);
            breakpoints.emplace_back(cost, bp_exit_soc);
        }

        // Add the breakpoints of the charging function for entry soc
        for (const auto& bp_of_phi : phi) {
            double bp_soc = bp_of_phi.image, bp_time = bp_of_phi.domain;
            double entry_soc = bp_soc;
            if (entry_soc <= min_entry_soc)
                continue;
            if (entry_soc >= max_entry_soc)
                break;
            double exit_soc = charge_for_time(phi, bp_soc, committed_charge_time);
            double cost     = prev_exit_soc.inverse(entry_soc) + fix_cost
                + (station_cost_profile.inverse(exit_soc) - station_cost_profile.inverse(entry_soc));
            breakpoints.emplace_back(cost, exit_soc);
        }

        // Add the breakpoints of the charging function for exit soc
        for (const auto& bp_of_phi : phi) {
            double bp_soc = bp_of_phi.image, bp_time = bp_of_phi.domain;
            if (bp_soc <= min_exit_soc)
                continue;
            if (bp_soc >= max_exit_soc)
                break;
            double entry_soc = initial_soc_when_charging(phi, bp_soc, committed_charge_time);
            double cost      = prev_exit_soc.inverse(entry_soc) + fix_cost
                + (station_cost_profile.inverse(bp_soc) - station_cost_profile.inverse(entry_soc));
            breakpoints.emplace_back(cost, bp_soc);
        }

        // Push max_entry_soc bp.
        {
            double max_entry_cost = prev_exit_soc.inverse(max_entry_soc);
            breakpoints.emplace_back(max_entry_cost + fix_cost
                    + (station_cost_profile.inverse(max_exit_soc) - station_cost_profile.inverse(max_entry_soc)),
                max_exit_soc);
        }

        std::sort(breakpoints.begin(), breakpoints.end());

        auto highest_point = std::max_element(
            breakpoints.begin(), breakpoints.end(), [](const auto& lhs, const auto& rhs) { return lhs.image < rhs.image; });
        if (highest_point != std::prev(breakpoints.end())) {
            breakpoints.erase(std::next(highest_point), breakpoints.end());
        }
        if (breakpoints.size() == 1) {
            breakpoints.push_back(breakpoints.back());
        }

        return construct_cost_profile_from_breakpoints(breakpoints, true, true);
    }

    CostProfile shift_by(const CostProfile& profile, double delta_cost, double delta_soc, double max_soc) {
        return frvcp::models::clip_image(frvcp::models::shift_by(profile, delta_cost, delta_soc), 0.0, max_soc);
    }

    bool is_flat(const CostProfile& profile) { return approx_eq(profile.getMaximumSoC(), profile.getMinimumSoC()); }
    const CostProfile::bp_container_t& get_breakpoints(const CostProfile& func) { return func.getBreakpoints(); }
    double value(const CostProfile& func, double x) { return func.value(x); }
    double inverse(const CostProfile& func, double y) { return func.inverse(y); }
    double get_lower_bound(const CostProfile& phi) { return phi.getMinimumCost(); }
    double get_upper_bound(const CostProfile& phi) { return phi.getMaximumCost(); }
    double get_image_lower_bound(const CostProfile& phi) { return phi.getMinimumSoC(); }
    double get_image_upper_bound(const CostProfile& phi) { return phi.getMaximumSoC(); }
}
