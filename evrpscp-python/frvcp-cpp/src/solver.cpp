#include "frvcp/solver.hpp"
#include "frvcp/label.hpp"
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <iostream>
#if ENABLE_PYTHON_BINDINGS
#include <fstream>
#include <pybind11/pybind11.h>
#include <pyerrors.h>
#endif

namespace {
#if ENABLE_PYTHON_BINDINGS
    void check_error_flags() {
        if (PyErr_CheckSignals() != 0) {
            std::cerr << "Detected signal sent to python!" << std::endl;
            throw pybind11::error_already_set();
        }
    }
#endif

    bool print_status_at_label(
        const frvcp::Label& label, frvcp::vertex_id extracted_at, const frvcp::Arc* propagated_arc = nullptr) {
        return false;
    }
}

namespace frvcp {

    Solver::Solver(const TimeExpandedNetwork& network)
        : _node_queue(cheapest_queued_label_comp { _label_queues })
        , _network(network) {
        _initialize();
    }

    std::pair<const label_t*, vertex_id> Solver::_extract_next_label() {
        vertex_id origin_vertex_id = _node_queue.extract_cheapest();
        auto settled_label         = _label_queues[origin_vertex_id].pop();
        const label_t* ptr         = settled_label.get();

        // Finally, settle the label
        _label_queues[origin_vertex_id].settle(std::move(settled_label));
        if (!_label_queues[origin_vertex_id].empty()) { // Reinsert vertex if the label queue is not empty
            _node_queue.insert(origin_vertex_id);
        }

        return { ptr, origin_vertex_id };
    }

    const label_t* Solver::solve(double ub) {
        /*{
            std::ofstream network("/tmp/network.gv");
            network << dump_as_dot(_Network());
        }*/
        //_label_queues[_Network().getSource()].add(std::make_unique<Label>());
        static const Label _root_label {};
        _label_queues[_Network().getSource()].add(&_root_label);
        _node_queue.insert(_Network().getSource());

        const label_t* best_solution = nullptr;

        while (!_node_queue.empty()) {
#if ENABLE_PYTHON_BINDINGS
            check_error_flags();
#endif
            auto [settled_label, origin_vertex_id] = _extract_next_label();

            if (settled_label->getMinimumCostAtSink() > ub && !is_root_label(*settled_label))
                continue;

            // Reached sink. Our algorithm is label setting, such that the first found solution should be optimal.
            if (origin_vertex_id == _Network().getSink()) {
                assert(settled_label->getNumberOfServedOperations() == _Network().getNumberOfOperations());
                // Feasible solution found
                if (best_solution != nullptr) {
                    if (certainly_gt(best_solution->getCostAtMinimumSoC(), settled_label->getCostAtMinimumSoC())) {
                        throw std::logic_error(
                            fmt::format("Algorithm is not label setting! Improved on BKS {} with new solution {}.",
                                best_solution->getCostAtMinimumSoC(), settled_label->getCostAtMinimumSoC()));
                    }
                } else {
                    best_solution = settled_label;
                }
                break;
            }

            // Propagate
            for (arc_id arc_index : _Network().getOutgoingArcs(origin_vertex_id)) {
                const Arc& arc         = _Network().getArc(arc_index);
                auto propagated_labels = propagate(*settled_label, arc, _Network().getVertex(origin_vertex_id),
                    _Network().getVertex(_Network().getTarget(arc_index)), arc_index);

                {
                    // Discard hash equal values
                    std::sort(propagated_labels.begin(), propagated_labels.end(),
                        [](const label_ref_t& lhs, const label_ref_t& rhs) { return lhs->hash() < rhs->hash(); });
                    auto new_end = std::unique(propagated_labels.begin(), propagated_labels.end(),
                        [](const label_ref_t& lhs, const label_ref_t& rhs) { return lhs->hash() == rhs->hash(); });
                    propagated_labels.erase(new_end, propagated_labels.end());

                    remove_dominated_labels(propagated_labels);
                }
                for (auto& label : propagated_labels) {
#ifdef ENABLE_CALLBACKS
                    if (_notify(_label_creation_callbacks, *label))
                        continue;
#endif
                    if (_label_queues[_Network().getTarget(arc_index)].add(std::move(label))) {
                        // Update node queue
                        _node_queue.update(_Network().getTarget(arc_index));
                    }
                }
            }
        }

        return best_solution;
    }

    void Solver::_validate_lower_bound(
        const label_t* best_solution) const { // std::cout << "Best sol: " << *best_solution << std::endl;
        const Label* prev = best_solution;
        bool end          = false;
        while (!is_root_label(*prev->getPredecessor())) {
            // Lower bound should hold
            if (certainly_gt(prev->getMinimumCostAtSink(), best_solution->getMinimumCostAtSink())) {
                std::cout << "Error LB does not hold for label" << *prev << '\n'
                          << " has LB of " << prev->getMinimumCostAtSink() << " but best solution has LB "
                          << best_solution->getMinimumCostAtSink()
                          << " delta: " << prev->getMinimumCostAtSink() - best_solution->getMinimumCostAtSink() << std::endl;
                end = true;
            }
            prev = prev->getPredecessor();
        }
        if (end) {
            dump_path(best_solution, &_Network());
            throw std::runtime_error("LB is not a lower bound!");
        }
    }

    void Solver::_initialize() {
        _node_queue.clear();
        _label_queues.resize(_Network().getNumberOfVertices());
        for (unsigned int i = 0; i < _Network().getNumberOfVertices(); ++i) {
            _label_queues[i].clear();
        }
    }

    const TimeExpandedNetwork& Solver::_Network() const { return _network.get(); }

    void Solver::setNetwork(const TimeExpandedNetwork& network) {
        _network = network;
        _initialize();
    }

    void Solver::reset() {
        _initialize();
        reset_label_allocator();
    }

#ifdef ENABLE_CALLBACKS
    void Solver::on_label_create(Solver::LabelCreationCallback cb) { _label_creation_callbacks.push_back(std::move(cb)); }
#endif
}