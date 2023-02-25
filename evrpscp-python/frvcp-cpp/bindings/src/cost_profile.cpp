#include "frvcp_bindings/cost_profile.hpp"
#include "frvcp_bindings/binding_helpers.hpp"
#include <frvcp/cost_profile.hpp>
#include <frvcp/models/battery.hpp>
#include <frvcp/models/charger.hpp>
#include <pybind11/stl.h>

using namespace frvcp;

void create_cost_profile_bindings(pybind11::module& m) {
    pybind11::class_<CostProfile>(m, "CostProfile")
        .def(pybind11::init<CostProfile::bp_container_t>())
        .def(pybind11::init<models::PWLFunction>())
        .def("getMinimumSoC", &CostProfile::getMinimumSoC)
        .def_property_readonly("minimum_soc", &CostProfile::getMinimumSoC)
        .def("getMaximumSoC", &CostProfile::getMaximumSoC)
        .def_property_readonly("maximum_soc", &CostProfile::getMaximumSoC)
        .def("getCostAtMinimumSoC", &CostProfile::getMinimumCost)
        .def_property_readonly("cost_at_minimum_soc", &CostProfile::getMinimumCost)
        .def("getMaximumCost", &CostProfile::getMaximumCost)
        .def_property_readonly("maximum_cost", &CostProfile::getMaximumCost)
        .def("value", &CostProfile::value)
        .def("inverse", &CostProfile::inverse)
        .def("__call__", &CostProfile::operator())
        .def("__eq__", &CostProfile::operator==)
        .def("__ne__", &CostProfile::operator!=)
        .def("__iter__", [](const CostProfile& profile){ return pybind11::make_iterator(profile.begin(), profile.end());}, pybind11::keep_alive<0, 1>())
        .def("__repr__", &ostream_to_string<CostProfile>);


    m.attr("INVALID_SOC") = pybind11::float_(INVALID_SOC);
    m.attr("COST_OF_UNREACHABLE_SOC") = pybind11::float_(COST_OF_UNREACHABLE_SOC);

    m.def("optimize_breakpoint_sequence", [](CostProfile f){ optimize_breakpoint_sequence(f); return f; }, "Optimize breakpoints of some cost profile");
    m.def("create_flat_profile", &create_flat_profile, pybind11::return_value_policy::automatic_reference);
    m.def("create_period_profile", &create_period_profile, pybind11::return_value_policy::automatic_reference);
    m.def("shift_by", &shift_by, pybind11::return_value_policy::automatic_reference);
    m.def("replace_station", &replace_station, pybind11::return_value_policy::automatic_reference);
    m.def("charge_at_intermediate_station", &charge_at_intermediate_station, pybind11::return_value_policy::automatic_reference);
}
