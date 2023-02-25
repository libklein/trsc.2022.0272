
#ifndef FRVCP_ALLOCATOR_HPP
#define FRVCP_ALLOCATOR_HPP

#include "frvcp/definitions.hpp"

#include <deque>
#include <memory>

namespace frvcp::util {

    template<class T> class Allocator {
        // static_assert(std::is_trivially_destructible_v<T>, "Allocator currently supports only trivially destructible
        // types!");
        using pool_type = std::unique_ptr<std::byte>;

    public:
        static constexpr size_t DEFAULT_POOL_SIZE = DEFAULT_ALLOCATOR_POOL_SIZE;
        inline static size_t deallocations        = 0;

    private:
        pool_type _pool;
        size_t _count = 0;
        size_t _size  = DEFAULT_POOL_SIZE;

    public:
        Allocator()
            : _pool(new std::byte[DEFAULT_POOL_SIZE * sizeof(T)]) {
            if (!_pool) {
                throw std::logic_error("Could not allocate label pool");
            }
        };
        explicit Allocator(size_t estimated_obj_count)
            : _pool(new std::byte[estimated_obj_count * sizeof(T)])
            , _size(estimated_obj_count) {
            if (!_pool) {
                throw std::logic_error("Could not allocate label pool");
            }
        };

        size_t size() const { return _size; }

        T* allocate() {
            if (_count >= _size) {
                throw std::runtime_error("Label slab exhausted!");
            }
            return std::addressof(reinterpret_cast<T*>(_pool.get())[_count++]);
        }

        void reset() {
            for (unsigned int i = 0; i < _count; ++i) {
                reinterpret_cast<T*>(_pool.get())[i].~T();
            }
            _count = 0;
        }

        size_t get_num_allocations() const { return _count; }

        void deallocate(T* ptr) {};

        void deallocate_last(T* ptr) {
            assert(reinterpret_cast<T*>(_pool.get()) + (_count - 1) == ptr);
            --_count;
            ptr->~T();
            ++deallocations;
        };

        ~Allocator() { reset(); };
    };

}

#endif // FRVCP_ALLOCATOR_HPP
