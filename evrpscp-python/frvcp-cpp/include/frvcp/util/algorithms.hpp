
#ifndef FRVCP_ALGORITHMS_HPP
#define FRVCP_ALGORITHMS_HPP

#include <utility>
#include <vector>

namespace frvcp::util {

    template<class ForwardIt, class BinaryPredicate>
    ForwardIt unique_keep_last(ForwardIt first, ForwardIt last, BinaryPredicate p) {
        if (first == last)
            return last;

        ForwardIt result = first;
        while (++first != last) {
            if (p(*result, *first)) {
                // *first will be overwritten in consecutive operations. Copy first to result to ensure that the last
                // seen value of *first is kept.
                *result = *first;
                continue;
            };

            if (++result != first) {
                *result = std::move(*first);
            }
        }
        return ++result;
    }

    template<class ForwardIt, class UnaryPredicate>
    std::vector<std::vector<typename ForwardIt::value_type>> split_if(ForwardIt first, ForwardIt last, UnaryPredicate p) {
        if (first == last) {
            return {};
        }
        std::vector<std::vector<typename ForwardIt::value_type>> splits(1);
        for (auto next = first; next != last; ++next) {
            if (p(*next)) {
                // Split
                std::vector<typename ForwardIt::value_type> next_cont {};
                splits.push_back(std::move(next_cont));
            }
            splits.back().push_back(std::move(*next));
        }

        return splits;
    }
}

#endif // FRVCP_ALGORITHMS_HPP
