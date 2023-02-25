#include "frvcp/models/battery.hpp"
#include <cassert>
#include <fmt/format.h>

namespace frvcp::models {

    Battery::Battery(
        WearCostDensityFunction wdf, double minimum_soc, double maximum_soc, double capacity, double initial_charge)
        : _wdf(std::move(wdf))
        , _minimum_soc(minimum_soc)
        , _maximum_soc(maximum_soc)
        , _capacity(capacity)
        , _initial_charge(initial_charge) {
        assert(getMinimumSoC() >= wdf.getMinimumSoC());
        assert(getMaximumSoC() <= wdf.getMaximumSoC());
    }

    double Battery::getMinimumSoC() const { return _minimum_soc; }
    double Battery::getMaximumSoC() const { return _maximum_soc; }
    double Battery::getBatteryCapacity() const { return _capacity; }
    double Battery::getInitialCharge() const { return _initial_charge; }

    const WearCostDensityFunction& Battery::getWDF() const { return _wdf; }

    double WearCostDensityFunction::getMinWearCost() const {
        throw std::runtime_error("Not implemented");
        return 0;
    }

    double WearCostDensityFunction::getWearCost(double from_soc, double to_soc) const {
        return value(to_soc) - value(from_soc);
    }

    WearCostDensityFunction::const_iterator WearCostDensityFunction::begin() {
        return PWLFunction::getBreakpoints().begin();
    }

    WearCostDensityFunction::const_iterator WearCostDensityFunction::end() { return PWLFunction::getBreakpoints().end(); }

    WearCostDensityFunction::const_iterator WearCostDensityFunction::begin() const {
        return PWLFunction::getBreakpoints().begin();
    }

    WearCostDensityFunction::const_iterator WearCostDensityFunction::end() const {
        return PWLFunction::getBreakpoints().end();
    }

    const PWLFunction::bp_container_t& WearCostDensityFunction::getBreakpoints() const {
        return PWLFunction::getBreakpoints();
    }

    double WearCostDensityFunction::getMinimumSoC() const { return getLowerBound(); }

    double WearCostDensityFunction::getMaximumSoC() const { return getUpperBound(); }

    double WearCostDensityFunction::getMaximumCost() const { return getImageUpperBound(); }

    double WearCostDensityFunction::getMinimumCost() const { return getImageLowerBound(); }

    std::ostream& operator<<(std::ostream& os, const WearCostDensityFunction& wdf) {
        os << "Wear cost density function: \n" << static_cast<const PWLFunction&>(wdf);
        return os;
    }

    WearCostDensityFunction::WearCostDensityFunction(PWLFunction::bp_container_t segments)
        : WearCostDensityFunction(PWLFunction(std::move(segments))) { }

    WearCostDensityFunction::WearCostDensityFunction(PWLFunction f)
        : PWLFunction(std::move(f)) {
        if (getLowerBound() != 0.0 || getImageLowerBound() != 0.0) {
            throw std::runtime_error(fmt::format(
                "WDF is malformed! SoC LB: {}, Cost LB: {}, concave: {}\n", getLowerBound(), getImageLowerBound()));
        }
        if (!is_convex(*this)) {
            throw std::runtime_error("Cannot construct non convex wear cost density function!");
        }
    }
}
