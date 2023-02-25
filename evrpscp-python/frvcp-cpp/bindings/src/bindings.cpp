#include <pybind11/pybind11.h>

#include <frvcp/definitions.hpp>

#include "frvcp_bindings/binding_helpers.hpp"

#include "frvcp_bindings/util.hpp"
#include "frvcp_bindings/pwl.hpp"
#include "frvcp_bindings/charger.hpp"
#include "frvcp_bindings/tour.hpp"
#include "frvcp_bindings/battery.hpp"
#include "frvcp_bindings/instance.hpp"
#include "frvcp_bindings/cost_profile.hpp"
#include "frvcp_bindings/network.hpp"
#include "frvcp_bindings/solver.hpp"
#include "frvcp_bindings/label.hpp"
#if false
#include "frvcp_bindings/node.hpp"
#include "frvcp_bindings/charging_schedule.hpp"
#include "frvcp_bindings/preprocessor.hpp"
#endif

PYBIND11_MODULE(evspnl, m) {
    m.attr("MAX_TOUR_ID") = pybind11::int_(frvcp::MAX_NUM_TOURS-1);
    m.attr("MAX_CHARGER_ID") = pybind11::int_(frvcp::MAX_NUM_CHARGERS-1);
    m.attr("PERIOD_LENGTH") = pybind11::float_(frvcp::PERIOD_LENGTH);
    // Utility
    create_utility_bindings(m);
    // PWL
    create_pwl_segment_bindings(m);
    create_pwl_function_bindings(m);
    // Battery
    create_wdf_bindings(m);
    create_battery_bindings(m);
    // Charger
    create_charging_function_binding(m);
    create_charger_binding(m);
    // Tour
    create_tour_bindings(m);
    // Instance
    create_instance_bindings(m);
    // Network bindings
    create_network_bindings(m);
    // Cost Profile
    create_cost_profile_bindings(m);
    // Labels
    create_label_bindings(m);
    // Solver
    create_solver_bindings(m);
}