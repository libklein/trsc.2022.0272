#include "frvcp_bindings/soc_profile.hpp"
#include <optional>
#include <pybind11/stl.h>

#include <frvcp/models/charger.hpp>
#include <frvcp/models/tour.hpp>
#include <frvcp/soc_profile.hpp>

using namespace frvcp;

void create_soc_profile_bindings(pybind11::module& m) {
    pybind11::class_<SoCProfile>(m, "SoCProfile")
        .def(pybind11::init<const models::Charger&, double, double>())
        .def("getMinSoC", &SoCProfile::getMinSoC, "get the SoC when not charging anything extra")
        .def("getMinTime", &SoCProfile::getMinTime, "get the minimum extra time required to remain feasible")
        .def("getMaxTime", &SoCProfile::getMaxTime, "get the maximum time chargeable")
        .def("getMaxSoC", &SoCProfile::getMaxSoC, "get the SoC when charging as much as possible")
        .def("getBreakpoints", [](const SoCProfile& profile){ return profile.getBreakpoints(); }, "get the breakpoints of the soc profile", pybind11::return_value_policy::reference_internal)
        .def("value", &SoCProfile::value, "get the soc when charging for an extra time of tau")
        .def("inverse", &SoCProfile::inverse, "get the time required to reach a soc of q")
        .def("propagate", pybind11::overload_cast<const models::Charger&, double>(&SoCProfile::propagate, pybind11::const_), "Propagates the SoC Profile to a charger and commits the specified time")
        .def("propagate", pybind11::overload_cast<const models::Tour&>(&SoCProfile::propagate, pybind11::const_), "Propagates the SoC Profile to a tour and travels it");
}
