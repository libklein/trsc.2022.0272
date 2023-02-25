#include "frvcp_bindings/battery.hpp"
#include "frvcp_bindings/binding_helpers.hpp"
#include <frvcp/models/battery.hpp>

using namespace frvcp::models;

void create_wdf_bindings(pybind11::module& m) {
    pybind11::class_<WearCostDensityFunction>(m, "WearCostDensityFunction")
        .def(pybind11::init<PWLFunction::bp_container_t>())
        .def(pybind11::init<PWLFunction>())
        .def("getMinimumSoC", &WearCostDensityFunction::getMinimumSoC, "Get the minimum SoC bound.")
        .def("getMaximumSoC", &WearCostDensityFunction::getMaximumSoC, "Get the maximum SoC bound.")
        .def("getMaximumCost", &WearCostDensityFunction::getMaximumCost, "Get the maximum cost reachable.")
        .def("getWearCost", &WearCostDensityFunction::getWearCost,
            "Get the wear cost incurred by charging from from_soc to to_soc.")
        .def_property_readonly("minimum_soc", &WearCostDensityFunction::getMinimumSoC, "Get the minimum SoC bound.")
        .def_property_readonly("maximum_soc", &WearCostDensityFunction::getMaximumSoC, "Get the maximum SoC bound.")
        .def_property_readonly("maximum_cost", &WearCostDensityFunction::getMaximumCost, "Get the maximum cost reachable.")
        .def(
            "__iter__", [](const WearCostDensityFunction& wdf) { return pybind11::make_iterator(wdf.begin(), wdf.end()); },
            pybind11::keep_alive<0, 1>())
        .def("__repr__", &ostream_to_string<WearCostDensityFunction>);
}

void create_battery_bindings(pybind11::module& m) {
    pybind11::class_<Battery>(m, "Battery")
        .def(pybind11::init<WearCostDensityFunction, double, double, double, double>())
        .def("getMinimumSoC", &Battery::getMinimumSoC, "Get the minimum SoC required.")
        .def("getMaximumSoC", &Battery::getMaximumSoC, "Get the maximum SoC required.")
        .def("getBatteryCapacity", &Battery::getBatteryCapacity, "Get the total battery capacity.")
        .def("getInitialCharge", &Battery::getInitialCharge, "Get the initial SoC.")
        .def("getWDF", &Battery::getWDF, "Get the wear-cost-density function of the battery.",
            pybind11::return_value_policy::reference_internal);
}