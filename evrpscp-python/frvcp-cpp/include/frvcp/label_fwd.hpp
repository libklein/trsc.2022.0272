
#ifndef FRVCP_LABEL_FWD_HPP
#define FRVCP_LABEL_FWD_HPP

#include <bitset>
#include <memory>

namespace frvcp {
    class Label;

    using label_t = Label;
    // using label_ref_t = std::unique_ptr<const label_t>;
    struct label_ref_wrapper {
        const label_t* _ptr;

        label_ref_wrapper(const label_t* ptr)
            : _ptr(ptr) {};

        const label_t& operator*() const { return *_ptr; }

        const label_t* operator->() const { return _ptr; }

        const label_t* get() const { return _ptr; }
    };
    using label_ref_t = label_ref_wrapper;
    // using label_ref_t = const label_t*;
}

#endif // FRVCP_LABEL_FWD_HPP
