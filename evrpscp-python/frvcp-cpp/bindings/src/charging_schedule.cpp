#include "frvcp_bindings/charging_schedule.hpp"

#include <pybind11/stl.h>

#include "frvcp_bindings/binding_helpers.hpp"
#include <frvcp/util/charging_schedule.hpp>

using namespace frvcp::util;

void create_schedule_bindings(pybind11::module &m) {
    pybind11::class_<operation>(m, "Operation")
        .def_property_readonly("node", [](const operation& op){ return op.node; })
        .def_readonly("begin", &operation::begin_time)
        .def_readonly("duration", &operation::duration)
        .def_readonly("entry_soc", &operation::entry_soc)
        .def_readonly("delta_soc", &operation::delta_soc)
        .def("__repr__", &ostream_to_string<operation>);

    pybind11::class_<charging_schedule>(m, "ChargingSchedule")
        .def_property_readonly("operations", &charging_schedule::getOperations)
        .def_property_readonly("cost", &charging_schedule::getCost)
        .def("isFeasible", &charging_schedule::isFeasible, "True if the schedule is feasible.")
        .def("__repr__", &ostream_to_string<charging_schedule>);
}