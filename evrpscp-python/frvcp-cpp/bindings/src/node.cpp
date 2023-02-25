#include "frvcp_bindings/node.hpp"
#include <frvcp/node.hpp>
#include <frvcp/models/charger.hpp>
#include <frvcp/models/tour.hpp>

using namespace frvcp;
using namespace frvcp::models;

void create_node_bindings(pybind11::module &m) {
    auto node_binding = pybind11::class_<Node>(m, "Node")
        .def(pybind11::init<size_t, Node::NodeType, double, double, double, double, double>())
        .def(pybind11::init<size_t, const Charger&, double, double, double, double>())
        .def(pybind11::init<size_t, const Tour&, double, double, double, double>())
        .def("getID", &Node::getID, "Get the node's id.")
        .def_property_readonly("isTour", [](const Node& node){ return node.getType() == frvcp::Node::TOUR; }, "True if the node is a tour")
        .def_property_readonly("isCharger", [](const Node& node){ return node.getType() == frvcp::Node::CHARGER; }, "True if the node is a charger")
        .def("getType", &Node::getType, "Get the node's type.")
        .def("getCharger", &Node::getCharger, "Get the node's charger (if it is a charger node).")
        .def("getTour", &Node::getTour, "Get the node's tour (if it is a tour node).")
        .def("getEnergyPrice", &Node::getEnergyPrice, "Get the node's energy price.")
        .def("getFixCost", &Node::getFixCost, "Get the node's fix cost.")
        .def("getBeginTime", &Node::getBeginTime, "Get the node's begin time.")
        .def("getEndTime", &Node::getEndTime, "Get the node's end time.")
        .def_property("fix_cost", &Node::getFixCost, &Node::setFixCost)
        .def_property("energy_price", &Node::getEnergyPrice, &Node::setEnergyPrice)
        .def_property("begin", &Node::getBeginTime, &Node::setBegin)
        .def_property("end", &Node::getEndTime, &Node::setEnd)
        .def_property("force_usage", &Node::getForceUsage, &Node::setForceUsage);

    pybind11::enum_<Node::NodeType>(node_binding, "NodeType")
        .value("TOUR", Node::NodeType::TOUR)
        .value("CHARGER", Node::NodeType::CHARGER)
        .value("SOURCE", Node::NodeType::SOURCE)
        .value("SINK", Node::NodeType::SINK);
}
