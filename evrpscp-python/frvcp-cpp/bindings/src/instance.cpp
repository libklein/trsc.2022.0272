#include "frvcp_bindings/instance.hpp"
#include <frvcp/models/instance.hpp>

using namespace frvcp::models;

void create_instance_bindings(pybind11::module &m) {
    pybind11::class_<Instance>(m, "Instance")
        .def(pybind11::init<Battery>())
        .def("addCharger", &Instance::addCharger, "Adds a new charger to the instance.", pybind11::return_value_policy::reference_internal)
        .def("addTour", &Instance::addTour, "Adds a new tour to the instance.", pybind11::return_value_policy::reference_internal)
        .def("getCharger", &Instance::getCharger, "Get the charger with the given id.", pybind11::return_value_policy::reference_internal)
        .def("getTour", &Instance::getTour, "Get the tour with the given id.", pybind11::return_value_policy::reference_internal)
        .def("getBattery", &Instance::getBattery, "Get the battery.", pybind11::return_value_policy::reference_internal)
        .def("getNumberOfChargers", &Instance::getNumberOfChargers, "Get the number of chargers registered.")
        .def("getNumberOfTours", &Instance::getNumberOfTours, "Get the number of tours registered.")
        .def("getTotalConsumption", &Instance::getTotalConsumption, "Get the total amount of charge required to serve all tours.");
}
