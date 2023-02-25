#ifndef FRVCP_LABEL_CONTAINER_HPP
#define FRVCP_LABEL_CONTAINER_HPP

#include "frvcp/label.hpp"
#include "frvcp/util/heap.hpp"
#include "ska/bytell_hash_map.hpp"
#include <deque>

namespace frvcp {
    namespace detail {
        inline bool _compare_deref_ptrs(const label_ref_t& lhs, const label_ref_t& rhs) { return *lhs > *rhs; };
    }

    class LabelQueue {
        // util::Heap<label_ref_t, decltype(&detail::_compare_deref_ptrs)> _unsettled_labels;
        struct _deref_ptrs_comp {
            bool operator()(const label_ref_t& lhs, const label_ref_t& rhs) { return *lhs > *rhs; }
        };
        // util::Heap<label_ref_t, decltype([](const label_ref_t& lhs, const label_ref_t& rhs){ return *lhs > *rhs; })>
        // _unsettled_labels;
        util::Heap<label_ref_t, _deref_ptrs_comp> _unsettled_labels;
        std::deque<label_ref_t> _settled_labels;
        ska::bytell_hash_set<std::size_t> _considered_labels;

    public:
        LabelQueue()                 = default;
        LabelQueue(LabelQueue&& rhs) = default;

        [[nodiscard]] bool empty() const;

        void clear();
        bool settle(label_ref_t label);

        /**
         * Adds the given label to the node queue if it is not dominated.
         * @param label The label handle
         * @return True if the label has been added, false if it was dominated and thus not added.
         */
        bool add(label_ref_t label);
        label_ref_t pop();
        [[nodiscard]] const label_t& top() const;
        decltype(_settled_labels.begin()) _find_dominator(const label_t& of_label);
    };
}

#endif // FRVCP_LABEL_CONTAINER_HPP
