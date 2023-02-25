
#ifndef FRVCP_LABEL_HPP
#define FRVCP_LABEL_HPP

#include "cost_profile.hpp"
#include "frvcp/label_fwd.hpp"
#include "frvcp/models/tour.hpp"
#include "frvcp/time_expanded_network.hpp"
#include <compare>
#include <vector>

#define CARRY_LABEL True

namespace frvcp {

    const Label* get_last_error_label();
    void dump_path(const Label* label, const TimeExpandedNetwork* network = nullptr);
    void reset_label_allocator();

    class Label {
    public:
        using tour_set_t = frvcp::tour_set_t;

    private:
        using arc_id = TimeExpandedNetwork::arc_id;

        const Label* _predecessor;
        const Label* _tracked_station;
        const arc_id _arc;

        CostProfile _charging_decisions;
        tour_set_t _served_operations;

        double _minimum_cost_at_sink;
        std::size_t _hash;

    public:
        Label();
        Label(const Label* predecessor, const arc_id arc_identifier, CostProfile charging_decisions,
            tour_set_t served_operations, double minimum_cost_at_sink);
        Label(const Label* predecessor, const Label* tracked_station, const arc_id arc_identifier,
            CostProfile charging_decisions, tour_set_t served_operations, double minimum_cost_at_sink);

#ifdef CARRY_LABEL
        std::size_t _id;
#endif
        const CostProfile& _profile() const { return _charging_decisions; };

    protected:
        std::size_t _propagate_service_arc(const Arc& arc, const Vertex& tail, const Vertex& head,
            const arc_id arc_identifer, std::vector<label_ref_t>&) const;
        std::size_t _propagate_charging_arc(const Arc& arc, const Vertex& tail, const Vertex& head,
            const arc_id arc_identifer, std::vector<label_ref_t>&) const;
        std::size_t _propagate_source_arc(const Arc& arc, const Vertex& tail, const Vertex& head,
            const arc_id arc_identifier, std::vector<label_ref_t>&) const;
        std::size_t _propagate_idling_arc(const Arc& arc, const Vertex& tail, const Vertex& head,
            const arc_id arc_identifier, std::vector<label_ref_t>&) const;

    public:
        [[nodiscard]] const Label* getPredecessor() const;
        [[nodiscard]] const Label* getTrackedStation() const;
        [[nodiscard]] const arc_id getArc() const;

        [[nodiscard]] double getMinimumCostAtSink() const;
        [[nodiscard]] double getCostAtMinimumSoC() const;
        [[nodiscard]] double getMinimumSoC() const;
        [[nodiscard]] double getMaximumSoC() const;

        [[nodiscard]] bool dominates(const Label& other) const;

        [[nodiscard]] bool servedTour(models::Tour::tour_id index) const;
        [[nodiscard]] std::size_t getNumberOfServedOperations() const;
        [[nodiscard]] const tour_set_t& getServedOperations() const;

        [[nodiscard]] std::size_t hash() const;

        std::partial_ordering operator<=>(const Label& other) const;
        bool operator==(const Label& rhs) const;
        bool operator!=(const Label& rhs) const;

        friend std::ostream& operator<<(std::ostream& out, const Label& label);
        friend std::vector<label_ref_t> propagate(
            const Label& label, const Arc& arc, const Vertex& tail, const Vertex& head, const arc_id arc_id);

        friend bool is_root_label(const Label& label);
        friend void dump_path(const Label* label, const TimeExpandedNetwork* network);

        template<class Iterator> friend bool is_set_dominated(const Label& label, Iterator begin, Iterator);

        bool is_increasing() const { return frvcp::models::is_increasing(_charging_decisions); }

        bool can_serve_tour() const {
            if (is_root_label(*this))
                return true;
            return this->_predecessor->getNumberOfServedOperations() == getNumberOfServedOperations();
        }
    };

    [[nodiscard]] bool replaces_station(const Label& label);
    [[nodiscard]] bool charges_intermediately(const Label& label);
    [[nodiscard]] bool provides_service(const Label& label);

    void remove_dominated_labels(std::vector<label_ref_t>& set_of_labels);
}

#endif // FRVCP_LABEL_HPP
