//
// Created by patrick on 23.10.18.
//

#ifndef FRVCP_BATTERY_HPP
#define FRVCP_BATTERY_HPP

#include "pwl.hpp"

namespace frvcp::models {

    /**
     * @brief Function mapping SoC to (accumulated) cost.
     * WearCostDensityFunctions provide a mapping of SoC to accumulated cost, i.e., give for any SoC q the cost incurred by
     * charging an initially empty battery to q.
     */
    class WearCostDensityFunction : private PWLFunction {
    public:
        using iterator       = PWLFunction::iterator;
        using const_iterator = PWLFunction::const_iterator;

        explicit WearCostDensityFunction(PWLFunction::bp_container_t segments);
        explicit WearCostDensityFunction(PWLFunction f);

        [[nodiscard]] double getMinimumSoC() const;
        [[nodiscard]] double getMaximumSoC() const;
        [[nodiscard]] double getMinimumCost() const;
        [[nodiscard]] double getMaximumCost() const;

        [[nodiscard]] const_iterator begin();
        [[nodiscard]] const_iterator end();
        [[nodiscard]] const_iterator begin() const;
        [[nodiscard]] const_iterator end() const;

        [[nodiscard]] const PWLFunction::bp_container_t& getBreakpoints() const;

        [[nodiscard]] double getWearCost(double from_soc, double to_soc) const;
        [[nodiscard]] double getMinWearCost() const;

        friend std::ostream& operator<<(std::ostream& os, const WearCostDensityFunction& wdf);
    };

    class Battery {
        /// Function mapping SoC to Cost
        WearCostDensityFunction _wdf;

        double _minimum_soc;
        double _maximum_soc;
        double _capacity;
        double _initial_charge;

    public:
        Battery(WearCostDensityFunction wdf, double minimum_soc, double maximum_soc, double capacity, double initial_charge);

        [[nodiscard]] double getMinimumSoC() const;
        [[nodiscard]] double getMaximumSoC() const;
        [[nodiscard]] double getBatteryCapacity() const;
        [[nodiscard]] double getInitialCharge() const;

        [[nodiscard]] const WearCostDensityFunction& getWDF() const;
    };

    double get_min_wear_cost(const Battery& battery);

}

#endif // FRVCP_BATTERY_HPP
