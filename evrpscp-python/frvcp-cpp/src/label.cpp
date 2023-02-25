#include "frvcp/label.hpp"

#include "frvcp/models/charger.hpp"
#include "frvcp/models/tour.hpp"

#include "frvcp/models/pwl_util.hpp"
#include "frvcp/time_expanded_network.hpp"
#include "frvcp/util/allocator.hpp"
#include "frvcp/util/debug.hpp"
#include "frvcp/util/floats.hpp"
#include <boost/container_hash/hash.hpp>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>

#include <cassert>
#include <iostream>
#include <vector>

#ifdef CARRY_LABEL
static std::size_t _next_label_id = 0;
#endif

static const frvcp::Label* LAST_ERROR_LABEL;

namespace frvcp {
    ::frvcp::util::Allocator<Label> _allocator;
}

namespace {
    using namespace frvcp;

    std::size_t compute_hash(const CostProfile& charging_decisions, std::size_t seed) {
        std::size_t hash_value = seed;
        for (const auto& bp : charging_decisions) {
            boost::hash_combine(hash_value, bp.domain);
            boost::hash_combine(hash_value, bp.image);
        }
        return hash_value;
    }

    double calculate_cost_lb(
        const Vertex& vertex, const tour_set_t& served_operations, const CostProfile& charging_decisions) {
        double remaining_consumption = vertex.getChargeNessesary(served_operations);
        // Lower bound on the cost required to recharge the charge required to traverse the rest of this tour.
        // We can either use a charger which we will visit in the future
        if (remaining_consumption > charging_decisions.getMinimumSoC()) {
            double cost_lb = charging_decisions.getMinimumCost()
                + vertex.getCostLowerBound(remaining_consumption - charging_decisions.getMinimumSoC());
            // Or the one we've visited earlier
            bool abort = false;
            for (const auto& bp : charging_decisions) {
                abort = bp.image > remaining_consumption;
                if (abort)
                    break;
                cost_lb = std::min(bp.domain + vertex.getCostLowerBound(remaining_consumption - bp.image), cost_lb);
            }
            // getMaximumSoC() > additional_charge_required
            if (abort) {
                cost_lb = std::min(charging_decisions.inverse(remaining_consumption), cost_lb);
            }
            assert(cost_lb >= charging_decisions.getMinimumCost());
            return cost_lb;
        }
        return charging_decisions.getMinimumCost();
    }

    // util::Allocator<Label> _allocator;

    template<class... Ts> label_ref_t allocate_label(Ts... params) {
        // return std::make_unique<label_t>(std::forward<Ts>(params)...);
        Label* allocated_label = frvcp::_allocator.allocate();
        new (allocated_label) Label(std::forward<Ts>(params)...);
        return label_ref_t(allocated_label);
    }

    bool check_pre_propagation_feasibility(const Label& label, const Arc& arc, const Vertex& tail) {
        double entry_soc = label.getMaximumSoC() + arc.getDeltaSoC();
        /*
         * A label is not feasible if one of the following holds:
         *      i) it does not reach the end of <arc> with a positive SoC
         *      ii) a operation that has to have been serviced already has not been serviced
         * Check ii) needs to be delayed for service arcs, as these may service the missing operation
         */
        if (!is_service_arc(arc)) {
            return entry_soc >= 0.0
                && (label.getServedOperations() & get_serviced_operations(tail)) == get_serviced_operations(tail);
        } else {
            return entry_soc >= 0.0;
        }
    }

    bool check_extended_pre_propagation_feasibility(
        const Label& label, const Arc& arc, [[maybe_unused]] const Vertex& tail) {
        // Serving a tour twice makes little sense.
        if (is_service_arc(arc) && label.servedTour(arc.getTour().getID())) {
            return false;
        }
        return true;
    }
}

namespace frvcp {
    const frvcp::Label* get_last_error_label() { return LAST_ERROR_LABEL; }

    void reset_label_allocator() { _allocator.reset(); }

    double Label::getCostAtMinimumSoC() const { return _charging_decisions.getMinimumCost(); }

    double Label::getMinimumSoC() const { return _charging_decisions.getMinimumSoC(); }

    double Label::getMaximumSoC() const { return _charging_decisions.getMaximumSoC(); }

    std::size_t Label::_propagate_charging_arc(const Arc& arc, const Vertex& tail, const Vertex& head,
        const arc_id arc_identifier, std::vector<label_ref_t>& spawned_labels) const {
        /*
         * Two possible actions:
         *  * Replacing the currently tracked station
         *  * Charging at intermediate station continuing to track the currently tracked station
         */
        const models::Charger& next_station = get_charger(tail);
        auto create_label = [this, head, arc_identifier](CostProfile profile, const Label* new_tracked_station) {
        // Compute LB - use tail here to ensure that charging opportunities at this vertex are captured.
#ifndef DO_NOT_USE_LB
            double cost_lb = calculate_cost_lb(head, this->getServedOperations(), profile);
#else
            double cost_lb = profile.getMinimumCost();
#endif
            return allocate_label(
                this, new_tracked_station, arc_identifier, std::move(profile), getServedOperations(), cost_lb);
        };
        /*
         * First - replace current station.
         * Each point c where
         *      1) replaced_slope(c-eps) < old_slope(c-eps)
         *      2) replaced_slope(c+eps) > old_slope(c+eps)
         * holds is non-dominated.
         * This is exactly the case when the old_slope (i.e. the slope of the old cost profile) decreases.
         * More precisely, as replaced_slope is concave, i.e., replaced_slope(c-eps) >= replaced_slope(c+eps), conditions 1)
         * and 2) cannot occur at breakpoints of replaced_slope. (A __change__ is not possible!) It could be worthwhile to
         * spawn a new label for each breakpoint of the current cost profile
         */
        const auto& station_cost_profile = get_charging_profile(tail);
        for (auto [cost, soc, slope] : _charging_decisions.getBreakpoints()) {
            double entry_cost    = cost + get_fix_cost(arc);
            double entry_soc     = soc + get_delta_soc(arc);
            double max_delta_soc = get_max_charge_delta(arc, next_station, entry_soc);
            max_delta_soc        = std::min(max_delta_soc, head.getChargeNessesary(_served_operations) - entry_soc);
            // Create a new label tracking this station
            spawned_labels.push_back(
                create_label(replace_station(station_cost_profile, max_delta_soc, entry_cost, entry_soc), nullptr));
        }

        /*
         * Second - continue tracking old station but charge at the current one.
         *
         * Only viable if charging at the new station is actually cheaper than charging at the old one. Otherwise
         * replacing is always better or at least as good.
         *
         * If the battery capacity or amount of charge remaining can be reached without charging at g, then we can skip
         * the procedure too.
         *
         * The only (potentially) worthwhile decision here is to charge as much as possible.
         * UB is the maximum charge or capacity
         */
        auto max_charge                 = tail.getCharger().getPhi().getMaximumSoC();
        double max_reasonable_soc_level = std::min(max_charge, tail.getChargeNessesary(this->getServedOperations()));
        if (_charging_decisions.getMaximumSoC() >= std::min(max_charge, tail.getChargeNessesary(this->getServedOperations()))
            // A flat profile does not capture any decisions and thus can be handled by station replacement,
            // simply committing to the single decision we have
            || is_flat(_charging_decisions)
            // We can just committ to charging as little as possible and replace the tracked station if intermediate charge
            // always yields a label which reaches the max reasonable soc
            || models::charge_for_time(tail.getCharger(), _charging_decisions.getMinimumSoC(), PERIOD_LENGTH)
                >= max_reasonable_soc_level) {
            return spawned_labels.size();
        }
        const CostProfile& tradeoff_at_tracked_station = _charging_decisions;
        const auto& phi_at_u                           = tail.getCharger().getPhi();

        auto calculated_tradeoffs_after_intermediate_charge = charge_intermediatly(tradeoff_at_tracked_station,
            station_cost_profile, phi_at_u, get_fix_cost(arc), PERIOD_LENGTH, max_reasonable_soc_level);

        std::transform(std::make_move_iterator(calculated_tradeoffs_after_intermediate_charge.begin()),
            std::make_move_iterator(calculated_tradeoffs_after_intermediate_charge.end()),
            std::back_inserter(spawned_labels), [&create_label, this](CostProfile spawned_profile) {
                return create_label(std::move(spawned_profile), _tracked_station);
            });

        return spawned_labels.size();
    }

    std::size_t Label::_propagate_source_arc(const Arc& arc, const Vertex& tail, [[maybe_unused]] const Vertex& head,
        const arc_id arc_identifier, std::vector<label_ref_t>& spawned_labels) const {
        auto shifted_profile
            = shift_by(this->_charging_decisions, get_fix_cost(arc), get_delta_soc(arc), get_delta_soc(arc));
#ifndef DO_NOT_USE_LB
        double cost_lb = shifted_profile.getMinimumCost()
            + tail.getCostLowerBound(tail.getChargeNessesary(getServedOperations()) - shifted_profile.getMinimumSoC());
#else
        double cost_lb = shifted_profile.getMinimumCost();
#endif
        spawned_labels.push_back(
            allocate_label(this, arc_identifier, std::move(shifted_profile), getServedOperations(), cost_lb));
        return 1;
    }

    std::size_t Label::_propagate_idling_arc([[maybe_unused]] const Arc& arc, [[maybe_unused]] const Vertex& tail,
        const Vertex& head, const arc_id arc_identifier, std::vector<label_ref_t>& spawned_labels) const {
        // Lower bound on the cost required to recharge the charge required to traverse the rest of this tour.
#ifndef DO_NOT_USE_LB
        double cost_lb = calculate_cost_lb(head, getServedOperations(), _charging_decisions);
#else
        double cost_lb = this->_charging_decisions.getMinimumCost();
#endif
        spawned_labels.push_back(
            allocate_label(this, arc_identifier, this->_charging_decisions, getServedOperations(), cost_lb));
        return 1;
    }

    std::size_t Label::_propagate_service_arc(const Arc& arc, const Vertex& tail, const Vertex& head,
        const arc_id arc_identifier, std::vector<label_ref_t>& spawned_labels) const {
        if (!can_serve_tour())
            return 0;
        auto propagated_served_operations = getServedOperations();
        propagated_served_operations.set(get_tour(arc).getID());
        // Additional feasibility check is nessesary here
        if ((propagated_served_operations & get_serviced_operations(tail)) == get_serviced_operations(tail)) {
            assert(get_delta_soc(arc) <= 0.0); // Otherwise max_soc is not set correctly
            double max_soc
                = std::min(this->_charging_decisions.getMaximumSoC(), head.getChargeNessesary(propagated_served_operations));
            auto shifted_profile = shift_by(this->_charging_decisions, get_fix_cost(arc), get_delta_soc(arc), max_soc);
#ifndef DO_NOT_USE_LB
            double cost_lb = calculate_cost_lb(head, propagated_served_operations, shifted_profile);
#else
            double cost_lb = shifted_profile.getMinimumCost();
#endif
            spawned_labels.push_back(allocate_label(
                this, arc_identifier, std::move(shifted_profile), std::move(propagated_served_operations), cost_lb));
            return 1;
        } else {
            return 0;
        }
    }

    Label::Label(const Label* predecessor, const arc_id arc_identifier, CostProfile charging_decisions,
        tour_set_t served_operations, double minimum_cost_at_sink)
        : Label(predecessor, predecessor->_tracked_station, arc_identifier, std::move(charging_decisions),
              std::move(served_operations), minimum_cost_at_sink) { }

    Label::Label(const Label* predecessor, const Label* tracked_station, const arc_id arc_identifier,
        CostProfile charging_decisions, tour_set_t served_operations, double minimum_cost_at_sink)
        : _predecessor(predecessor)
        , _tracked_station(tracked_station != nullptr ? tracked_station : this)
        , _arc(arc_identifier)
        , _charging_decisions(std::move(charging_decisions))
        , _served_operations(std::move(served_operations))
        , _minimum_cost_at_sink(minimum_cost_at_sink)
        , _hash(compute_hash(_charging_decisions, boost::hash_value(_served_operations.to_ulong()))) {
#ifdef CARRY_LABEL
        _id = _next_label_id++;
#endif
    }

    std::vector<label_ref_t> propagate(const Label& label, const Arc& arc, const Vertex& tail, const Vertex& head,
        const TimeExpandedNetwork::arc_id arc_identifier) {
        std::vector<label_ref_t> created_labels;
        // Do basic feasibility check first
        if (!check_pre_propagation_feasibility(label, arc, head))
            return created_labels;
        // Then extended feasibility check
        if (!check_extended_pre_propagation_feasibility(label, arc, head))
            return created_labels;
        switch (arc.getType()) {
        case Arc::SERVICE_ARC: label._propagate_service_arc(arc, tail, head, arc_identifier, created_labels); break;
        case Arc::CHARGING_ARC: label._propagate_charging_arc(arc, tail, head, arc_identifier, created_labels); break;
        case Arc::SOURCE_ARC: label._propagate_source_arc(arc, tail, head, arc_identifier, created_labels); break;
        case Arc::IDLE_ARC: label._propagate_idling_arc(arc, tail, head, arc_identifier, created_labels); break;
        default: throw std::runtime_error("Unknown arc type");
        }

        return created_labels;
    }

    Label::Label()
        : _predecessor(nullptr)
        , _tracked_station(nullptr)
        , _charging_decisions(create_flat_profile(0.0, 0.0))
        , _served_operations()
        , _minimum_cost_at_sink(0.0)
        , _hash(compute_hash(_charging_decisions, boost::hash_value(_served_operations.to_ulong())))
#ifdef CARRY_LABEL
        , _id(_next_label_id++)
#endif
    {
    }

    const arc_id Label::getArc() const { return _arc; }

    bool Label::servedTour(models::Tour::tour_id index) const { return _served_operations.test(index); }

    std::size_t Label::getNumberOfServedOperations() const { return _served_operations.count(); }

    const tour_set_t& Label::getServedOperations() const { return _served_operations; }

    std::partial_ordering Label::operator<=>(const Label& other) const {
        auto min_cost_comp = getCostAtMinimumSoC() <=> other.getCostAtMinimumSoC();
        if (min_cost_comp == std::partial_ordering::equivalent) {
            if (auto min_soc_comp = getMinimumSoC() <=> other.getMinimumSoC();
                min_cost_comp == std::partial_ordering::equivalent) {
                if (auto max_cost_comp = _charging_decisions.getMaximumCost() <=> other._charging_decisions.getMaximumCost();
                    max_cost_comp == std::partial_ordering::equivalent) {
                    return _charging_decisions.getMaximumSoC() <=> other._charging_decisions.getMaximumSoC();
                } else {
                    return max_cost_comp;
                }
            } else {
                return min_soc_comp;
            }
        }
        return min_cost_comp;
    }

    bool Label::dominates(const Label& other) const {
        if (other.can_serve_tour() && !this->can_serve_tour()) {
            return false;
        }
        // We cannot dominate other if other serves some operations we do not service
        if ((other.getServedOperations() | getServedOperations()) != getServedOperations()) {
            return false;
        }
        // If other has lower minimum cost we cannot dominate it (other.value(other.getMinimumCost()) > -infity ==
        // this->value(other.getMinimumCost())
        if (other.getCostAtMinimumSoC() < getCostAtMinimumSoC()) {
            return false;
        }
        // If other reaches a higher SoC we cannot dominate it
        if (other._charging_decisions.getMaximumSoC() > _charging_decisions.getMaximumSoC()) {
            return false;
        }

        for (const auto& other_bp : other._charging_decisions) {
            if (other_bp.image > _charging_decisions.value(other_bp.domain))
                return false;
        }

        for (const auto& this_bp : _charging_decisions) {
            if (this_bp.image < other._charging_decisions.value(this_bp.domain))
                return false;
        }

        return true;
    }

    std::ostream& operator<<(std::ostream& out, const Label& label) {
        out << "[Label ";
#ifdef CARRY_LABEL
        out << label._id << " ";
#endif
        if (replaces_station(label)) {
            out << "(replace) ";
        } else if (charges_intermediately(label)) {
            out << "(intermediate) ";
        } else if (provides_service(label)) {
            out << "(service) ";
        } else {
            out << "(idle) ";
        }
        out << "with lb " << label.getMinimumCostAtSink() << ", cost " << label.getCostAtMinimumSoC() << " at "
            << label.getArc() << " tracking ";
#ifdef CARRY_LABEL
        out << (label.getTrackedStation() != nullptr ? label.getTrackedStation()->_id : 1);
#else
        out << label.getTrackedStation();
#endif
        out << " serves " << label.getServedOperations() << " profile " << label._charging_decisions << "]" << std::flush;
        return out;
    }

    bool is_root_label(const Label& label) { return label._predecessor == nullptr; }

    void dump_path(const Label* label, const TimeExpandedNetwork* network) {
        while (!is_root_label(*(label = label->_predecessor))) {
            if (network != nullptr) {
                fmt::print("\t{} at ({}, {})\n", *label, network->getVertex(network->getOrigin(label->getArc())),
                    network->getVertex(network->getTarget(label->getArc())));
            } else {
                fmt::print("\t{}\n", *label);
            }
        }
    }

    const Label* Label::getPredecessor() const { return _predecessor; }

    const Label* Label::getTrackedStation() const { return _tracked_station; }

    double Label::getMinimumCostAtSink() const { return _minimum_cost_at_sink; }
    bool Label::operator==(const Label& rhs) const {
        return _predecessor == rhs._predecessor && _tracked_station == rhs._tracked_station && _arc == rhs._arc
            && _charging_decisions == rhs._charging_decisions && _served_operations == rhs._served_operations
            && _minimum_cost_at_sink == rhs._minimum_cost_at_sink && _id == rhs._id;
    }

    bool Label::operator!=(const Label& rhs) const { return !(rhs == *this); }

    template<class Iterator> bool is_set_dominated(const Label& label, Iterator begin, Iterator end) {
        if (begin == end)
            return false;
        if (certainly_lt(label.getCostAtMinimumSoC(), (*std::min_element(begin, end, [](const Label* lhs, const Label* rhs) {
                return lhs->getCostAtMinimumSoC() < rhs->getCostAtMinimumSoC();
            }))->getCostAtMinimumSoC())) {
            return false;
        }
        if (certainly_gt(label.getMaximumSoC(), (*std::max_element(begin, end, [](const Label* lhs, const Label* rhs) {
                return lhs->getMaximumSoC() < rhs->getMaximumSoC();
            }))->getMaximumSoC())) {
            return false;
        }

        auto pareto_front_value = [begin, end](double cost) {
            double highest_soc = std::numeric_limits<double>::lowest();
            for (auto beg = begin; beg != end; ++beg) {
                if (double this_soc = (*beg)->_charging_decisions.value(cost); this_soc > highest_soc) {
                    highest_soc = this_soc;
                }
            }
            return highest_soc;
        };

        const auto& label_cost_profile = label._charging_decisions;
        for (const auto& bp : label_cost_profile) {
            if (certainly_gt(bp.image, pareto_front_value(bp.domain)))
                return false;
        }
        for (auto next_profile = begin; next_profile != end; ++next_profile) {
            for (const auto& bp : (*next_profile)->_charging_decisions) {
                if (certainly_gt(label_cost_profile.value(bp.domain), bp.image))
                    return false;
            }
        }

        return true;
    }

    std::size_t Label::hash() const { return _hash; }

    bool replaces_station(const Label& label) { return label.getTrackedStation() == &label; }

    bool charges_intermediately(const Label& label) {
        return !is_root_label(label) && !replaces_station(label)
            && label.getPredecessor()->getMinimumSoC() < label.getMinimumSoC();
    }

    bool provides_service(const Label& label) {
        if (!label.getPredecessor()) {
            return false;
        }
        return label.getNumberOfServedOperations() > label.getPredecessor()->getNumberOfServedOperations();
    }

    void remove_dominated_labels(std::vector<label_ref_t>& set_of_labels) {
        std::vector<const Label*> label_ptrs(set_of_labels.size());
        std::transform(set_of_labels.begin(), set_of_labels.end(), label_ptrs.begin(),
            [](const label_ref_t& copy_candidate) { return copy_candidate.get(); });

        auto remove_from_container = [](auto& container, auto pos) {
            std::iter_swap(container.end() - 1, pos);
            container.pop_back();
            return pos;
        };

        auto remove_label = [&label_ptrs, &set_of_labels, &remove_from_container](auto label_iter) {
            auto index = std::distance(set_of_labels.begin(), label_iter);
            remove_from_container(label_ptrs, label_ptrs.begin() + index);
            return remove_from_container(set_of_labels, label_iter);
        };

        for (auto next_label = set_of_labels.begin(); next_label != set_of_labels.end();) {
            // Create set without next_label
            std::vector<const Label*> reduced_labels(label_ptrs.size() - 1);
            std::copy_if(label_ptrs.begin(), label_ptrs.end(), reduced_labels.begin(),
                [next_label](const Label* ptr) { return ptr != next_label->get(); });

            if (is_set_dominated(**next_label, reduced_labels.begin(), reduced_labels.end())) {
                next_label = remove_label(next_label);
            } else {
                ++next_label;
            }
        }
    }
}
