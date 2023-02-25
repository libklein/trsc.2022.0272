#include "frvcp/definitions.hpp"
#include "frvcp_bindings/label.hpp"
#include "frvcp_bindings/binding_helpers.hpp"
#include <frvcp/label.hpp>

using namespace frvcp;
using namespace frvcp::models;

void create_label_bindings(pybind11::module& m) {
    pybind11::class_<Label>(m, "Label")
        .def(pybind11::init())
        .def(pybind11::init([](const Label* label, const arc_id arc_id, CostProfile profile, tour_set_t served_tours){
            double cost = profile.getMinimumCost(); // Not a sequence point
            return Label(label, arc_id, std::move(profile), served_tours, cost);
        }))
        .def(pybind11::init<const Label*, const arc_id, CostProfile, tour_set_t, double>())
        .def_property_readonly("arc", &Label::getArc, "Get the id of the generating arc of this label.")
        .def_property_readonly("minimum_cost", &Label::getCostAtMinimumSoC, "Get the minimum cost at the target vertex.")
        .def_property_readonly("cost_lb", &Label::getMinimumCostAtSink, "Get the minimum cost at the sink vertex.")
        .def_property_readonly("minimum_soc", &Label::getMinimumSoC, "Get the minimum SoC at the target vertex.")
        .def_property_readonly("maximum_soc", &Label::getMaximumSoC, "Get the maximum reachable SoC at the target vertex.")
        .def_property_readonly("number_of_served_operations", &Label::getNumberOfServedOperations)
        .def_property_readonly("served_operations", [](const Label& label){
            std::vector<tour_id> served_ops;
            for(tour_id i = 0; i < MAX_NUM_TOURS; ++i) {
                if(label.servedTour(i)) {
                    served_ops.push_back(i);
                }
            }
            return served_ops;
        }, pybind11::return_value_policy::move)
        .def_property_readonly("served_operations_bitset", [](const Label& label) {
            return label.getServedOperations().to_ulong();
        })
        .def("dominates", &Label::dominates, "Check whether this label dominates the passed label.")
        .def("servedTour", &Label::servedTour, "Check whether this label dominates the passed label.")
        .def("__repr__", &ostream_to_string<Label>)
        .def_property_readonly("is_root", [](const Label& label) { return is_root_label(label); })
        .def_property_readonly("predecessor", &Label::getPredecessor)
        .def_property_readonly("tracked_station", &Label::getTrackedStation)
        .def_property_readonly("intermediate_charge", [](const Label& label) { return charges_intermediately(label); })
        .def_property_readonly("replaces_station", [](const Label& label) { return replaces_station(label); })
        .def_property_readonly("provides_service", [](const Label& label) { return provides_service(label); })
        .def_property_readonly("_profile", &Label::_profile);

    m.def("get_last_error_label", &get_last_error_label);
}
