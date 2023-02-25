#ifndef FRVCP_SOLVER_HPP
#define FRVCP_SOLVER_HPP

#include "frvcp/label.hpp"
#include "frvcp/time_expanded_network.hpp"
#include "frvcp/util/label_container.hpp"
#include "frvcp/util/node_queue.hpp"
#include <vector>

namespace frvcp {

    class Solver {
    public:
        using LabelCreationCallback = std::function<bool(const Label&)>;

    private:
        struct cheapest_queued_label_comp {
            const std::vector<LabelQueue>& _label_queues;
            explicit cheapest_queued_label_comp(const std::vector<LabelQueue>& label_queues)
                : _label_queues(label_queues) {};

            bool operator()(vertex_id lhs, vertex_id rhs) { return _label_queues[lhs].top() < _label_queues[rhs].top(); }
        };

        using vertex_id = TimeExpandedNetwork::vertex_id;
        using arc_id    = TimeExpandedNetwork::arc_id;
        NodeQueue<vertex_id, cheapest_queued_label_comp> _node_queue;
        std::vector<LabelQueue> _label_queues;

        std::reference_wrapper<const TimeExpandedNetwork> _network;

#ifdef ENABLE_CALLBACKS
        std::vector<LabelCreationCallback> _label_creation_callbacks;
#endif
    protected:
        void _initialize();

        const TimeExpandedNetwork& _Network() const;

    private:
        /**
         * Extracts the currently cheapest label and performs nessesary maintenance, i.e., settles it and updates the node
         * queue.
         * @return A pointer to the (now settled) label and the id of the vertex it was extracted at.
         */
        std::pair<const label_t*, vertex_id> _extract_next_label();

#ifdef ENABLE_CALLBACKS
        template<class Callbacks, class... Args> bool _notify(Callbacks& callback_list, Args&&... args) {
            return std::any_of(callback_list.begin(), callback_list.end(),
                [&args...](LabelCreationCallback& callback) { return callback(std::forward<Args>(args)...); });
        }
#endif
    public:
        explicit Solver(const TimeExpandedNetwork& network);

        void setNetwork(const TimeExpandedNetwork& network);
        void reset();

#ifdef ENABLE_CALLBACKS
        void on_label_create(LabelCreationCallback callback);
#endif

        /**
         * Solves a SPPRC on the given network.
         * @param ub Upper cost threshold - any path which exceeds these costs will be discarded.
         * @return nullptr if no solution was found. Otherwise a pointer to the label representing the cheapest path to the
         * sink.
         */
        const label_t* solve(double ub = 0.0);
        void _validate_lower_bound(const label_t* best_solution) const;
    };
}

#endif // FRVCP_SOLVER_HPP
