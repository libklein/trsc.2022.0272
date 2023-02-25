#include "frvcp/time_expanded_network.hpp"
#include "frvcp/models/charger.hpp"
#include "frvcp/models/pwl.hpp"
#include "frvcp/models/tour.hpp"

#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/isomorphism.hpp>
#include <cassert>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <iostream>

namespace frvcp {

    Vertex::Vertex(
        Period period, double energy_price, std::shared_ptr<const models::Charger> charger, CostProfile charging_decisions)
        : _type(Vertex::VertexType::STATION)
        , _begin_of_period(period)
        , _energy_price(energy_price)
        , _charger(std::move(charger))
        , _charging_decisions(std::move(charging_decisions))
        , _potentially_required_charge(0.0)
        , _lowest_cost_to_charge(models::create_constant_pwl(0.0, std::numeric_limits<double>::max(), 0.0)) {
#ifdef ENABLE_SAFETY_CHECKS
        if (!_charger) {
            throw std::runtime_error("Cannot instantiate a station vertex of without specifing a charger");
        }
        _check_input_data();
#endif
    }

    Vertex::Vertex(Vertex::VertexType type, Period period, double energy_price)
        : _type(type)
        , _begin_of_period(period)
        , _energy_price(energy_price)
        , _charging_decisions(create_flat_profile(0.0, 0.0))
        , _potentially_required_charge(0.0)
        , _lowest_cost_to_charge(models::create_constant_pwl(0.0, std::numeric_limits<double>::max(), 0.0)) {
#ifdef ENABLE_SAFETY_CHECKS
        if (is_station(*this)) {
            throw std::runtime_error("Cannot instantiate a station vertex of without specifing a charger");
        }
        _check_input_data();
#endif
    }

    void Vertex::_check_input_data() {
        if (_energy_price < 0.0) {
            throw std::runtime_error("Negative energy prices are not supported!");
        }
    }

    Vertex::VertexType Vertex::getType() const { return _type; }

    void Vertex::setServicedOperation(std::size_t index, bool set) { _required_tours.set(index, set); }

    Period Vertex::getPeriod() const { return _begin_of_period; }

    double Vertex::getEnergyPrice() const { return _energy_price; }

    const models::Charger& Vertex::getCharger() const {
#ifdef ENABLE_SAFETY_CHECKS
        if (!is_station(*this)) {
            throw std::runtime_error("Cannot get charger of non-station vertex!");
        }
#endif
        return *_charger;
    }

    const CostProfile& Vertex::getChargingDecisions() const {
#ifdef ENABLE_SAFETY_CHECKS
        if (!is_station(*this)) {
            std::cerr << "[WARN]: Trying to get profile of non-station vertex!" << std::endl;
        }
#endif
        return _charging_decisions;
    }
    const tour_set_t& Vertex::getCompletedOperations() const { return _required_tours; }

    bool Vertex::hasBeenCompleted(std::size_t index) const { return _required_tours.test(index); }

    bool Vertex::operator==(const Vertex& rhs) const {
        return _type == rhs._type && _begin_of_period == rhs._begin_of_period && _energy_price == rhs._energy_price
            && _required_tours == rhs._required_tours && _potentially_required_charge == rhs._potentially_required_charge
            && _charger == rhs._charger && _charging_decisions == rhs._charging_decisions
            && _potentially_remaining_operations == rhs._potentially_remaining_operations;
    }

    bool Vertex::operator!=(const Vertex& rhs) const { return !(rhs == *this); }
    std::ostream& operator<<(std::ostream& os, const Vertex& vertex) {
        os << "(P" << vertex.getPeriod() << " T";
        switch (vertex.getType()) {
        case Vertex::STATION: os << "C" << get_charger(vertex).getID(); break;
        case Vertex::SOURCE: os << "O"; break;
        case Vertex::SINK: os << "S"; break;
        case Vertex::GARAGE: os << "G"; break;
        }
        os << ")[" << vertex.getEnergyPrice() << "]";
        return os;
    }

    void Vertex::addPotentiallyUnservicedOperation(const std::shared_ptr<const models::Tour>& tour) {
        _potentially_remaining_operations.push_back(tour);
        _potentially_required_charge += tour->getConsumption();
    }

    double Vertex::getChargeNessesary(const tour_set_t& serviced_operations) const {
        double nessesary_charge = _potentially_required_charge;
        for (const auto& tour : _potentially_remaining_operations) {
            nessesary_charge
                -= serviced_operations.test(tour->getID()) * !hasBeenCompleted(tour->getID()) * tour->getConsumption();
        }
#ifdef ENABLE_SAFETY_CHECKS
        if (nessesary_charge < 0.0) {
            throw std::runtime_error(
                fmt::format("Vertex {} has nessesary charge of {} < 0 for serviced ops {}. Pot. req. mem: {}", *this,
                    nessesary_charge, serviced_operations, _potentially_required_charge));
        }
#endif
        return nessesary_charge;
    }

    double Vertex::getCostLowerBound(double delta_soc) const {
        if (delta_soc <= 0.0) {
            return 0.0;
        } else if (delta_soc > _lowest_cost_to_charge.getUpperBound()) {
#ifdef ENABLE_SAFETY_CHECKS
#ifdef WARN_EXTENSIVE_SAFETY_CHECKS
            fmt::print("[WARNING]: Called getCostLowerBound with unreachable delta soc {} at {}", delta_soc, *this);
#endif
            throw std::runtime_error(
                fmt::format("Called getCostLowerBound with unreachable delta soc {} at {}", delta_soc, *this));
#endif
            return COST_OF_UNREACHABLE_SOC;
        }

        double l = _lowest_cost_to_charge.value(delta_soc);
#ifdef ENABLE_SAFETY_CHECKS
        if (std::isnan(l)) {
            for (const auto& bp : _lowest_cost_to_charge.getBreakpoints()) {
                fmt::print("{}\n", bp);
            }
            throw std::runtime_error(
                fmt::format("Cost profile at {} yields {} for soc {}: {}.", *this, l, delta_soc, _lowest_cost_to_charge));
        }
#endif
        return l;
    }

    void Vertex::setLowestCostToCharge(models::PWLFunction lowest_cost_to_charge) {
        _lowest_cost_to_charge = std::move(lowest_cost_to_charge);
    }

    Arc::Arc(Arc::ArcType type, double fix_cost, double soc_consumption, Period duration,
        std::shared_ptr<const models::Tour> service_operation)
        : _type(type)
        , _fix_cost(fix_cost)
        , _delta_soc(soc_consumption)
        , _duration_periods(duration)
        , _service_operation(std::move(service_operation)) {
#ifdef ENABLE_SAFETY_CHECKS
        if (is_service_arc(*this) && !_service_operation) {
            throw std::runtime_error("Cannot create service arc without assigned service operation!");
        }
        if (_service_operation && _service_operation->getID() >= MAX_NUM_TOURS) {
            throw std::runtime_error("Cannot create arc! Tour ID exceeds maximum supported tour id!");
        }
        if (duration < 1) {
            throw std::runtime_error(fmt::format("Cannot create arc with duration {} < 1!", duration));
        }
#endif
    }

    Arc::Arc(double fix_cost, double soc_consumption, Period duration, std::shared_ptr<const models::Tour> service_operation)
        : Arc(ArcType::SERVICE_ARC, fix_cost, soc_consumption, duration, std::move(service_operation)) { }

    Arc::Arc(Arc::ArcType type, double fix_cost, double soc_consumption, Period duration)
        : Arc(type, fix_cost, soc_consumption, duration, nullptr) { }

    Arc::ArcType Arc::getType() const { return _type; }

    double Arc::getFixCost() const { return _fix_cost; }

    double Arc::getDeltaSoC() const { return _delta_soc; }

    Period Arc::getDuration() const { return _duration_periods; }

    const models::Tour& Arc::getTour() const {
#ifdef ENABLE_SAFETY_CHECKS
        if (!is_service_arc(*this)) {
            throw std::runtime_error("Cannot get tour of non service arc!");
        }
#endif
        return *_service_operation;
    }

    void Arc::setFixCost(double pi) { _fix_cost = pi; }
    bool Arc::operator==(const Arc& rhs) const {
        return _type == rhs._type && _fix_cost == rhs._fix_cost && _delta_soc == rhs._delta_soc
            && _duration_periods == rhs._duration_periods && _service_operation == rhs._service_operation;
    }
    bool Arc::operator!=(const Arc& rhs) const { return !(rhs == *this); }
    std::ostream& operator<<(std::ostream& os, const Arc& arc) {
        os << "(" << get_fix_cost(arc) << "$ " << arc.getDeltaSoC() << "q ";
        switch (arc.getType()) {
        case Arc::SOURCE_ARC: os << "O"; break;
        case Arc::CHARGING_ARC: os << "C"; break;
        case Arc::SERVICE_ARC: os << "S " << get_tour(arc).getID() << " :" << get_duration_time(arc); break;
        case Arc::IDLE_ARC: os << "I"; break;
        }
        os << ")";
        return os;
    }

    const Vertex& TimeExpandedNetwork::getVertex(TimeExpandedNetwork::vertex_id id) const { return _graph[id]; }

    const Arc& TimeExpandedNetwork::getArc(TimeExpandedNetwork::arc_id id) const { return _graph[id]; }

    TimeExpandedNetwork::vertex_id TimeExpandedNetwork::getOrigin(TimeExpandedNetwork::arc_id id) const {
        return boost::source(id, _graph);
    }

    TimeExpandedNetwork::vertex_id TimeExpandedNetwork::getTarget(TimeExpandedNetwork::arc_id id) const {
        return boost::target(id, _graph);
    }

    TimeExpandedNetwork::vertex_id TimeExpandedNetwork::getSource() const { return _source; }

    TimeExpandedNetwork::vertex_id TimeExpandedNetwork::getSink() const { return _sink; }

    std::size_t TimeExpandedNetwork::getNumberOfVertices() const { return boost::num_vertices(_graph); }

    std::size_t TimeExpandedNetwork::getNumberOfArcs() const { return boost::num_edges(_graph); }

    std::size_t TimeExpandedNetwork::getNumberOfOperations() const {
        return get_serviced_operations(getVertex(getSink())).count();
    }

    auto TimeExpandedNetwork::getOutgoingArcs(vertex_id vertex) const -> out_edges {
        auto [out_edge_begin, out_edge_end] = boost::out_edges(vertex, _graph);
        return { out_edge_begin, out_edge_end };
    }

    TimeExpandedNetwork::arc_id TimeExpandedNetwork::addArc(
        TimeExpandedNetwork::vertex_id origin, TimeExpandedNetwork::vertex_id target, const Arc& arc) {
#ifdef ENABLE_SAFETY_CHECKS
        if (origin == target) {
            throw std::runtime_error(fmt::format("Trying to add loop ({})-({}) to time expanded network", origin, target));
        }
#endif

        if (is_service_arc(arc)) {
            _graph[getSink()].setServicedOperation(get_tour(arc).getID(), true);
        }

        return boost::add_edge(origin, target, arc, _graph).first;
    }

    TimeExpandedNetwork::vertex_id TimeExpandedNetwork::addVertex(const Vertex& vertex) {
        return boost::add_vertex(vertex, _graph);
    }

    std::string dump_as_dot(const TimeExpandedNetwork& network) {
        class DotVisitor : public boost::default_bfs_visitor {
            const TimeExpandedNetwork& _network;
            std::stringstream& _out;

        public:
            DotVisitor(const TimeExpandedNetwork& network, std::stringstream& out)
                : _network(network)
                , _out(out) {
                _out << "digraph {\n";
            }

            void examine_vertex(TimeExpandedNetwork::graph_t::vertex_descriptor v, const TimeExpandedNetwork::graph_t& g) {
                _out << '\t' << v << dump_as_dot(_network.getVertex(v)) << ";\n";
            }

            void examine_edge(TimeExpandedNetwork::graph_t::edge_descriptor e, const TimeExpandedNetwork::graph_t& g) {
                _out << '\t' << _network.getOrigin(e) << " -> " << _network.getTarget(e) << dump_as_dot(_network.getArc(e))
                     << ";\n";
            }
        };

        std::stringstream dot_representation;
        boost::breadth_first_search(
            network._graph, network.getSource(), boost::visitor(DotVisitor(network, dot_representation)));
        // We could also use finalize_vertex(u, g) where u == sink, but then we could not print unconnected graphs
        // which may be useful for debugging purposes.
        dot_representation << "}";
        return dot_representation.str();
    }
    TimeExpandedNetwork::arcs TimeExpandedNetwork::getArcs() const {
        auto [beg, end] = boost::edges(_graph);
        return { beg, end };
    }
    TimeExpandedNetwork::vertices TimeExpandedNetwork::getVertices() const {
        auto [beg, end] = boost::vertices(_graph);
        return { beg, end };
    }

    TimeExpandedNetwork::TimeExpandedNetwork()
        : _source(addVertex(Vertex(Vertex::VertexType::SOURCE, 0, 0.0)))
        , _sink(addVertex(Vertex(Vertex::VertexType::SINK, std::numeric_limits<Period>::max(), 0.0))) { }

    bool TimeExpandedNetwork::operator==(const TimeExpandedNetwork& other) const {
        if (getNumberOfOperations() != other.getNumberOfOperations()) {
            std::cout << "Wrong number of ops!" << std::endl;
            return false;
        };

        // Vertex/Arc count check is handled by boost
        std::vector<graph_t::vertex_descriptor> iso_map(getNumberOfVertices());
        if (!boost::isomorphism(_graph, other._graph,
                boost::isomorphism_map(boost::make_iterator_property_map(
                    iso_map.begin(), boost::get(boost::vertex_index, _graph), iso_map.front())))) {
            std::cout << "Not isomorphic!" << std::endl;
            return false;
        }
        // Check for arc/vertex property equality

        class EqVisitor : public boost::default_bfs_visitor {
            const TimeExpandedNetwork::graph_t& _other;
            const std::vector<graph_t::vertex_descriptor>& _iso_map;
            bool& _result;

        public:
            EqVisitor(const TimeExpandedNetwork::graph_t& other, decltype(_iso_map)& iso_map, bool& result)
                : _other(other)
                , _iso_map(iso_map)
                , _result(result) {};

            void examine_vertex(TimeExpandedNetwork::graph_t::vertex_descriptor v, const TimeExpandedNetwork::graph_t& g) {
                std::cout << boost::vertex(v, g) << " == " << boost::vertex(_iso_map[v], _other)
                          << "?: " << (boost::vertex(v, g) == boost::vertex(_iso_map[v], _other)) << std::endl;
                _result &= boost::vertex(v, g) == boost::vertex(_iso_map[v], _other);
            }

            void examine_edge(TimeExpandedNetwork::graph_t::edge_descriptor e, const TimeExpandedNetwork::graph_t& g) {
                _result &= boost::edge(e.m_source, e.m_target, g)
                    == boost::edge(_iso_map[e.m_source], _iso_map[e.m_target], _other);
            }
        };

        bool is_equal = true;
        boost::breadth_first_search(_graph, _source, boost::visitor(EqVisitor(other._graph, iso_map, is_equal)));
        std::cout << "Graph equals? " << is_equal << std::endl;
        return is_equal;
    }

    void TimeExpandedNetwork::removeVertex(vertex_id id) { boost::remove_vertex(id, _graph); }

    void TimeExpandedNetwork::removeArc(arc_id id) { boost::remove_edge(id, _graph); }

    std::string dump_as_dot(const Vertex& vertex) { return fmt::format("[label=\"{}\"]", vertex); }

    std::string dump_as_dot(const Arc& arc) {
        std::stringstream attr;
        attr << fmt::format("[label=\"pi={} dq={}\" ", arc.getFixCost(), arc.getDeltaSoC());
        if (is_service_arc(arc)) {
            attr << "color=\"blue\" style=\"dashed\" ";
        }
        attr << "]";
        return attr.str();
    }

    bool is_station(const Vertex& vertex) { return vertex.getType() == Vertex::VertexType::STATION; }
    bool is_source(const Vertex& vertex) { return vertex.getType() == Vertex::VertexType::SOURCE; }
    bool is_sink(const Vertex& vertex) { return vertex.getType() == Vertex::VertexType::SINK; }
    bool is_garage(const Vertex& vertex) { return vertex.getType() == Vertex::VertexType::GARAGE; }
    bool is_charging_arc(const Arc& arc) { return arc.getType() == Arc::ArcType::CHARGING_ARC; }
    bool is_service_arc(const Arc& arc) { return arc.getType() == Arc::ArcType::SERVICE_ARC; }
    bool is_source_arc(const Arc& arc) { return arc.getType() == Arc::ArcType::SOURCE_ARC; }
    bool is_idle_arc(const Arc& arc) { return arc.getType() == Arc::ArcType::IDLE_ARC; }
    const models::Charger& get_charger(const Vertex& vertex) { return vertex.getCharger(); }
    double get_energy_price(const Vertex& vertex) { return vertex.getEnergyPrice(); }
    const CostProfile& get_charging_profile(const Vertex& vertex) { return vertex.getChargingDecisions(); }
    tour_set_t get_serviced_operations(const Vertex& vertex) { return vertex.getCompletedOperations(); }
    const models::Tour& get_tour(const Arc& arc) { return arc.getTour(); }
    Period get_duration(const Arc& arc) { return arc.getDuration(); }
    double get_duration_time(const Arc& arc) { return arc.getDuration() * PERIOD_LENGTH; }
    double get_delta_soc(const Arc& arc) { return arc.getDeltaSoC(); }
    double get_fix_cost(const Arc& arc) { return arc.getFixCost(); }
    double get_max_charge_delta(const Arc& arc, const models::Charger& charger, double entry_soc) {
        return charger.getPhi().getCharge(entry_soc, get_duration_time(arc)) - entry_soc;
    }
    Period get_period(const Vertex& vertex) { return vertex.getPeriod(); }
}