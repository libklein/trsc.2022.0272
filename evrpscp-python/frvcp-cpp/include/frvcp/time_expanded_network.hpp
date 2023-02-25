
#ifndef FRVCP_TIME_EXPANDED_NETWORK_HPP
#define FRVCP_TIME_EXPANDED_NETWORK_HPP

#include "frvcp/cost_profile.hpp"
#include "frvcp/models/fwd.hpp"
#include <boost/graph/adjacency_list.hpp>
#include <ostream>

namespace frvcp {
    class Vertex {
        /**
         * Can be
         *  * Station
         *  * Source
         *  * Sink
         *  * Idle
         */
    public:
        enum VertexType { STATION, SOURCE, SINK, GARAGE };
        Vertex(Period period, double energy_price, std::shared_ptr<const models::Charger> charger,
            CostProfile charging_decisions);
        Vertex(VertexType type, Period period, double energy_price);
        Vertex() = default;

    private:
        VertexType _type;
        Period _begin_of_period;
        double _energy_price;
        std::shared_ptr<const models::Charger> _charger;
        CostProfile _charging_decisions;

        tour_set_t _required_tours;
        double _potentially_required_charge;
        std::vector<std::shared_ptr<const models::Tour>> _potentially_remaining_operations;
        models::PWLFunction _lowest_cost_to_charge
            = models::create_constant_pwl(0.0, std::numeric_limits<double>::max(), 0.0);

        void _check_input_data();

    public:
        [[nodiscard]] VertexType getType() const;

        void setServicedOperation(std::size_t index, bool set = true);
        void addPotentiallyUnservicedOperation(const std::shared_ptr<const models::Tour>& tour);
        void setLowestCostToCharge(models::PWLFunction lowest_cost_to_charge);

        /**
         * @return The period which's begin this vertex marks.
         */
        [[nodiscard]] Period getPeriod() const;
        [[nodiscard]] double getEnergyPrice() const;
        [[nodiscard]] const models::Charger& getCharger() const;
        [[nodiscard]] const CostProfile& getChargingDecisions() const;
        [[nodiscard]] const tour_set_t& getCompletedOperations() const;
        [[nodiscard]] bool hasBeenCompleted(std::size_t index) const;
        [[nodiscard]] double getChargeNessesary(const tour_set_t& serviced_operations) const;
        [[nodiscard]] double getCostLowerBound(double delta_soc) const;

        [[nodiscard]] bool operator==(const Vertex& rhs) const;
        [[nodiscard]] bool operator!=(const Vertex& rhs) const;

        friend std::ostream& operator<<(std::ostream& os, const Vertex& vertex);
    };

    class Arc {
    public:
        enum ArcType { SOURCE_ARC, CHARGING_ARC, SERVICE_ARC, IDLE_ARC };

    private:
        ArcType _type;

        std::shared_ptr<const models::Tour> _service_operation;
        double _fix_cost;
        double _delta_soc;
        Period _duration_periods;

        Arc(ArcType type, double fix_cost, double soc_consumption, Period duration,
            std::shared_ptr<const models::Tour> service_operation);

    public:
        Arc(double fix_cost, double soc_consumption, Period duration, std::shared_ptr<const models::Tour> service_operation);
        Arc(ArcType type, double fix_cost, double soc_consumption, Period duration);

        [[nodiscard]] ArcType getType() const;

        [[nodiscard]] double getFixCost() const;
        [[nodiscard]] double getDeltaSoC() const;
        [[nodiscard]] Period getDuration() const;

        void setFixCost(double fix_cost);
        [[nodiscard]] const models::Tour& getTour() const;

        bool operator==(const Arc& rhs) const;
        bool operator!=(const Arc& rhs) const;
        friend std::ostream& operator<<(std::ostream& os, const Arc& arc);
    };

    bool is_station(const Vertex& vertex);
    bool is_source(const Vertex& vertex);
    bool is_sink(const Vertex& vertex);
    bool is_garage(const Vertex& vertex);

    bool is_charging_arc(const Arc& arc);
    bool is_service_arc(const Arc& arc);
    bool is_source_arc(const Arc& arc);
    bool is_idle_arc(const Arc& arc);

    const models::Charger& get_charger(const Vertex& vertex);
    double get_energy_price(const Vertex& vertex);
    Period get_period(const Vertex& vertex);
    const CostProfile& get_charging_profile(const Vertex& vertex);
    /**
     *
     * @param vertex
     * @return The operations that any feasible path has to service before this point in time/space.
     */
    tour_set_t get_serviced_operations(const Vertex& vertex);

    const models::Tour& get_tour(const Arc& arc);
    Period get_duration(const Arc& arc);
    double get_duration_time(const Arc& arc);
    double get_delta_soc(const Arc& arc);
    double get_fix_cost(const Arc& arc);
    double get_max_charge_delta(const Arc& arc, const models::Charger& charger, double entry_soc);

    class TimeExpandedNetwork {
        using graph_t = boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, Vertex, Arc>;

        template<class iterator> struct iter_wrapper {
            iterator _begin, _end;

            iterator begin() const { return _begin; };
            iterator end() const { return _end; };
        };

    public:
        using vertex_id    = graph_t::vertex_descriptor;
        using arc_id       = graph_t::edge_descriptor;
        using operation_id = std::size_t;
        using charger_id   = std::size_t;
        using out_edges    = iter_wrapper<graph_t::out_edge_iterator>;
        using vertices     = iter_wrapper<graph_t::vertex_iterator>;
        using arcs         = iter_wrapper<graph_t::edge_iterator>;

    private:
        graph_t _graph;
        vertex_id _source;
        vertex_id _sink;

    public:
        TimeExpandedNetwork();

        vertex_id addVertex(const Vertex& vertex);
        arc_id addArc(vertex_id origin, vertex_id target, const Arc& arc);

        [[nodiscard]] const Vertex& getVertex(vertex_id id) const;
        [[nodiscard]] const Arc& getArc(arc_id id) const;

        void removeVertex(vertex_id id);
        void removeArc(arc_id id);

        [[nodiscard]] vertex_id getOrigin(arc_id id) const;
        [[nodiscard]] vertex_id getTarget(arc_id id) const;

        [[nodiscard]] vertex_id getSource() const;
        [[nodiscard]] vertex_id getSink() const;

        [[nodiscard]] std::size_t getNumberOfVertices() const;
        [[nodiscard]] std::size_t getNumberOfArcs() const;

        [[nodiscard]] std::size_t getNumberOfOperations() const;

        [[nodiscard]] out_edges getOutgoingArcs(vertex_id vertex) const;
        [[nodiscard]] arcs getArcs() const;
        [[nodiscard]] vertices getVertices() const;

        [[nodiscard]] bool operator==(const TimeExpandedNetwork& other) const;

        friend std::string dump_as_dot(const TimeExpandedNetwork& network);
    };

    using vertex_id = TimeExpandedNetwork::vertex_id;
    using arc_id    = TimeExpandedNetwork::arc_id;
    static_assert(std::is_integral_v<vertex_id>);

    std::string dump_as_dot(const TimeExpandedNetwork& network);
    std::string dump_as_dot(const Vertex& vertex);
    std::string dump_as_dot(const Arc& arc);
}

#endif // FRVCP_TIME_EXPANDED_NETWORK_HPP
