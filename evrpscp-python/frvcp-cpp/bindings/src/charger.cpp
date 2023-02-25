#include "frvcp_bindings/charger.hpp"
#include "frvcp_bindings/binding_helpers.hpp"
#include <frvcp/models/charger.hpp>

using namespace frvcp::models;

void create_charging_function_binding(pybind11::module& m) {
    pybind11::class_<ChargingFunction>(m, "ChargingFunction")
        .def(pybind11::init<PWLFunction::bp_container_t>())
        .def(pybind11::init<PWLFunction>())
        .def("getMinimumSoC", &ChargingFunction::getMinimumSoC)
        .def_property_readonly("minimum_soc", &ChargingFunction::getMinimumSoC)
        .def("getMaximumSoC", &ChargingFunction::getMaximumSoC)
        .def_property_readonly("maximum_soc", &ChargingFunction::getMaximumSoC)
        .def("getFullChargeDuration", &ChargingFunction::getFullChargeDuration)
        .def_property_readonly("full_charge_duration", &ChargingFunction::getFullChargeDuration)
        .def("getCharge", &ChargingFunction::getCharge)
        .def("getDuration", &ChargingFunction::getDuration)
        .def("soc_after", &ChargingFunction::getSoCAfter)
        .def("time_required", &ChargingFunction::getTimeRequired)
        .def(
            "__iter__", [](const ChargingFunction& phi) { return pybind11::make_iterator(phi.begin(), phi.end()); },
            pybind11::keep_alive<0, 1>())
        .def("__repr__", &ostream_to_string<ChargingFunction>);
}

void create_charger_binding(pybind11::module& m) {
    pybind11::class_<Charger, std::shared_ptr<Charger>>(m, "Charger")
        .def(pybind11::init(
            [](ChargingFunction phi, Charger::charger_id id) { return std::make_shared<Charger>(std::move(phi), id); }))
        .def("getID", &Charger::getID, "Returns true if the charger is a base charger")
        .def("getPhi", &Charger::getPhi, "Get the charger's charging function",
            pybind11::return_value_policy::reference_internal)
        .def_property_readonly("id", &Charger::getID, "Returns true if the charger is a base charger")
        .def_property_readonly("phi", &Charger::getPhi, "Get the charger's charging function",
            pybind11::return_value_policy::reference_internal);
}