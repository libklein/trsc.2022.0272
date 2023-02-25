#include "frvcp/util/label_container.hpp"
#include "frvcp/label.hpp"

#include <algorithm>

#include <fmt/format.h>
#include <fmt/ostream.h>

namespace frvcp {

    bool LabelQueue::empty() const { return _unsettled_labels.empty(); }

    void LabelQueue::clear() {
        _unsettled_labels.clear();
        _settled_labels.clear();
    }

    bool LabelQueue::settle(label_ref_t label) {
        auto insert_at = std::upper_bound(_settled_labels.begin(), _settled_labels.end(), label,
            [](const label_ref_t& lhs, const label_ref_t& rhs) { return lhs->getMaximumSoC() < rhs->getMaximumSoC(); });
        _settled_labels.insert(insert_at, std::move(label));
        return true;
    }

    bool LabelQueue::add(label_ref_t label) {
        if (auto [iter, success] = _considered_labels.insert(label->hash()); !success) {
            return false;
        }

        if (_unsettled_labels.empty()) {
            if (auto dominator = _find_dominator(*label); dominator != _settled_labels.end()) {
                return false;
            }
        } else {
            const auto& prev_top = _unsettled_labels.top();
            if (_deref_ptrs_comp {}(prev_top, label)) {
                // Label replaces prev_top
                if (auto dominator = _find_dominator(*label); dominator != _settled_labels.end()) {
                    return false;
                }
            }
        }
        // If there exist unsettled labels, it's safe to push the label on the stack without checking dominance -
        // if the pushed label is dominated that will be detected when it becomes the top. If it is the new top,
        // it replaces the old top and thus cannot be dominated (it's cheaper)
        _unsettled_labels.push(std::move(label));

        return true;
    }

    label_ref_t LabelQueue::pop() {
        auto extracted_label = _unsettled_labels.pop();
        // Ensure that the new top of the heap is not dominated
        while (!empty()) {
            const auto& next_cheapest_label = _unsettled_labels.top();
            if (auto dominator = _find_dominator(*next_cheapest_label);
                dominator == _settled_labels.end() && !extracted_label->dominates(*next_cheapest_label)) {
                break;
            }
            _unsettled_labels.pop();
        }

        return extracted_label;
    }

    decltype(LabelQueue::_settled_labels.begin()) LabelQueue::_find_dominator(const label_t& of_label) {
        // Get first label for which !(settled->getMaximumSoC() < of_label->getMaximumSoC()) -> settled reaches a higher max
        // soc
        auto dominator = std::lower_bound(_settled_labels.begin(), _settled_labels.end(), of_label,
            [](const label_ref_t& settled_label, const label_t& val) {
                return settled_label->getMaximumSoC() < val.getMaximumSoC();
            });
        for (; dominator != _settled_labels.end(); ++dominator) {
            if ((*dominator)->dominates(of_label)) {
                break;
            }
        }
        return dominator;
    }

    const label_t& LabelQueue::top() const { return *_unsettled_labels.top(); }
}