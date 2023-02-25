#include "frvcp_bindings/tour.hpp"
#include "frvcp_bindings/binding_helpers.hpp"
#include <frvcp/models/tour.hpp>

using namespace frvcp;
using namespace frvcp::models;

void create_tour_bindings(pybind11::module &m) {
    pybind11::class_<Tour, std::shared_ptr<Tour>>(m, "ServiceOperation")
        .def(pybind11::init([](Tour::tour_id id, double consumption, Period earliest_departure, Period latest_departure, Period duration){
            return std::make_shared<Tour>(id, consumption, earliest_departure, latest_departure, duration);
        }))
        .def_property("id", &Tour::getID, &Tour::setID, "Get the service operation's id.")
        .def_property_readonly("consumption", &Tour::getConsumption, "Get the service operations's consumption.")
        .def_property_readonly("duration", &Tour::getDuration, "Get the service operation's duration.")
        .def_property_readonly("earliest_departure", &Tour::getEarliestDeparturePeriod, "Get the service operation's earliest departure period.")
        .def_property_readonly("latest_departure", &Tour::getLatestDeparturePeriod, "Get the service operation's latest departure period.")
        .def("getID", &Tour::getID, "Get the service operation's id.")
        .def("getDeltaSoC", &Tour::getConsumption, "Get the service operations's consumption.")
        .def("getDuration", &Tour::getDuration, "Get the service operation's duration.")
        .def("getLatestDeparturePeriod", &Tour::getLatestDeparturePeriod, "Get the service operation's latest departure period.")
        .def("getEarliestDeparturePeriod", &Tour::getEarliestDeparturePeriod, "Get the service operation's earliest departure period.")
        .def("__repr__", &ostream_to_string<Tour>);
}
