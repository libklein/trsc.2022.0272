
#ifndef FRVCP_NODE_QUEUE_HPP
#define FRVCP_NODE_QUEUE_HPP

#include <functional>
#include <stdexcept>

namespace frvcp {
    template<class T, class Comparator = std::less<T>> class NodeQueue {
        Comparator _compare;

        using container_t = std::vector<T>;
        container_t _container;

    public:
        NodeQueue() = default;
        explicit NodeQueue(Comparator comp);

        bool empty() const { return _container.empty(); }

        T extract_cheapest() {
#ifdef ENABLE_SAFETY_CHECKS
            if (empty()) {
                throw std::runtime_error("Cannot extract from empty container!");
            }
#endif
            auto min_elem = std::min_element(_container.begin(), _container.end(), _compare);

            std::iter_swap(min_elem, std::prev(_container.end()));
            T elem = std::move(_container.back());
            _container.pop_back();

            return elem;
        }

        void update(const T& elem) {
            auto pos = std::find(_container.begin(), _container.end(), elem);
            if (pos == _container.end()) {
                insert(elem);
            }
        }

        void insert(T elem) { _container.push_back(std::move(elem)); }

        void clear() { _container.clear(); }
    };

    template<class T, class Comparator>
    NodeQueue<T, Comparator>::NodeQueue(Comparator comp)
        : _compare(std::move(comp)) { }
}

#endif // FRVCP_NODE_QUEUE_HPP
