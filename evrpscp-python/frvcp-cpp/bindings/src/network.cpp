#include "frvcp_bindings/network.hpp"
#include <frvcp/time_expanded_network.hpp>


#include <frvcp/models/tour.hpp>
#include <frvcp/models/charger.hpp>

using namespace frvcp;
using namespace frvcp::models;

namespace {
    void create_vertex_bindings(pybind11::module& m) {
        auto _vertex_binding = pybind11::class_<Vertex>(m, "Vertex");

        pybind11::enum_<Vertex::VertexType>(m, "VertexType")
            .value("Source", Vertex::VertexType::SOURCE)
            .value("Sink", Vertex::VertexType::SINK)
            .value("Garage", Vertex::VertexType::GARAGE)
            .value("Station", Vertex::VertexType::STATION)
            .export_values();

        _vertex_binding
            .def(pybind11::init<Period, double, std::shared_ptr<const models::Charger>, CostProfile>())
            .def(pybind11::init<Vertex::VertexType, Period, double>())
            .def_property_readonly("period", &get_period)
            .def_property_readonly("type", &Vertex::getType)
            .def_property_readonly("energy_price", &get_energy_price)
            .def_property_readonly("charger", &get_charger, pybind11::return_value_policy::reference_internal)
            .def_property_readonly("charging_decisions", &get_charging_profile, pybind11::return_value_policy::reference_internal)
            .def_property_readonly("is_station", &is_station)
            .def_property_readonly("is_source", &is_source)
            .def_property_readonly("is_sink", &is_sink)
            .def_property_readonly("is_garage", &is_garage)
            .def("set_service_required", &Vertex::setServicedOperation)
            .def("is_service_required", &Vertex::hasBeenCompleted)
            .def("add_potentially_unserviced_operation", &Vertex::addPotentiallyUnservicedOperation)
            .def("set_soc_cost_lower_bound", &Vertex::setLowestCostToCharge)
            .def("__eq__", &Vertex::operator==);
    }

    void create_arc_bindings(pybind11::module& m) {
        auto _arc_binding = pybind11::class_<Arc>(m, "Arc");

        pybind11::enum_<Arc::ArcType>(m, "ArcType")
            .value("Source", Arc::ArcType::SOURCE_ARC)
            .value("Idle", Arc::ArcType::IDLE_ARC)
            .value("Service", Arc::ArcType::SERVICE_ARC)
            .value("Charging", Arc::ArcType::CHARGING_ARC)
            .export_values();


        _arc_binding
            .def(pybind11::init<double, double, Period, std::shared_ptr<const models::Tour>>())
            .def(pybind11::init<Arc::ArcType, double, double, Period>())
            .def_property_readonly("type", &Arc::getType)
            .def_property_readonly("is_charging_arc", &is_charging_arc)
            .def_property_readonly("is_service_arc", &is_service_arc)
            .def_property_readonly("is_source_arc", &is_source_arc)
            .def_property_readonly("is_idle_arc", &is_idle_arc)
            .def_property("fix_cost", &get_fix_cost, &Arc::setFixCost)
            .def_property_readonly("delta_soc", &get_delta_soc)
            .def_property_readonly("duration", &get_duration)
            .def_property_readonly("duration_time", &get_duration_time)
            .def_property_readonly("service_operation", &get_tour, pybind11::return_value_policy::reference_internal)
            .def("__eq__", &Arc::operator==);

        m.def("get_max_charge_delta", get_max_charge_delta);
    }
}

void create_network_bindings(pybind11::module &m) {
    create_vertex_bindings(m);
    create_arc_bindings(m);

    pybind11::class_<TimeExpandedNetwork>(m, "TimeExpandedNetwork")
        .def(pybind11::init<>())
        .def("add_vertex", &TimeExpandedNetwork::addVertex)
        .def("add_arc", &TimeExpandedNetwork::addArc)
        .def("remove_vertex", &TimeExpandedNetwork::removeVertex)
        .def("remove_arc", &TimeExpandedNetwork::removeArc)
        .def("get_vertex", &TimeExpandedNetwork::getVertex, pybind11::return_value_policy::reference_internal)
        .def("get_arc", &TimeExpandedNetwork::getArc, pybind11::return_value_policy::reference_internal)
        .def("get_origin", &TimeExpandedNetwork::getOrigin)
        .def("get_target", &TimeExpandedNetwork::getTarget)
        .def_property_readonly("source", &TimeExpandedNetwork::getSource)
        .def_property_readonly("sink", &TimeExpandedNetwork::getSink)
        .def_property_readonly("number_of_vertices", &TimeExpandedNetwork::getNumberOfVertices)
        .def_property_readonly("number_of_arcs", &TimeExpandedNetwork::getNumberOfArcs)
        .def_property_readonly("number_of_operations", &TimeExpandedNetwork::getNumberOfOperations)
        .def("get_outgoing_arcs", [](const TimeExpandedNetwork& network, TimeExpandedNetwork::vertex_id v){ return pybind11::make_iterator(network.getOutgoingArcs(v).begin(), network.getOutgoingArcs(v).end());}, pybind11::keep_alive<0, 1>())
        .def_property_readonly("vertices", [](const TimeExpandedNetwork& network){ return pybind11::make_iterator(network.getVertices().begin(), network.getVertices().end());}, pybind11::keep_alive<0, 1>())
        .def("__iter__", [](const TimeExpandedNetwork& network){ return pybind11::make_iterator(network.getVertices().begin(), network.getVertices().end());}, pybind11::keep_alive<0, 1>())
        .def_property_readonly("arcs", [](const TimeExpandedNetwork& network){ return pybind11::make_iterator(network.getArcs().begin(), network.getArcs().end());}, pybind11::keep_alive<0, 1>())
        .def("__repr__", pybind11::overload_cast<const TimeExpandedNetwork&>(&dump_as_dot))
        .def("__eq__", &TimeExpandedNetwork::operator==);

    pybind11::class_<TimeExpandedNetwork::arc_id>(m, "ArcID")
        .def_property("origin", [](const TimeExpandedNetwork::arc_id& id) { return id.m_source; }, [](TimeExpandedNetwork::arc_id& id, TimeExpandedNetwork::vertex_id v) { id.m_source = v; })
        .def_property("target", [](const TimeExpandedNetwork::arc_id& id) { return id.m_target; }, [](TimeExpandedNetwork::arc_id& id, TimeExpandedNetwork::vertex_id v) { id.m_target = v; })
        .def("__eq__", [](TimeExpandedNetwork::arc_id a, TimeExpandedNetwork::arc_id b){ return a == b; })
        .def("__hash__", [](const arc_id& id){ return boost::hash<arc_id>{}(id); });

    pybind11::class_<TimeExpandedNetwork::vertex_id>(m, "VertexID")
        .def("__eq__", [](const vertex_id& lhs, const vertex_id& rhs){ return lhs == rhs; })
        .def("__hash__", [](const vertex_id& vid){ return std::hash<vertex_id>{}(vid); });
}