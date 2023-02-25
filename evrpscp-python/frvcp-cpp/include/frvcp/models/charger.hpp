
#ifndef FRVCP_CHARGER_HPP
#define FRVCP_CHARGER_HPP

#include "pwl.hpp"

#include <memory>
#include <ostream>

namespace frvcp::models {
    class ChargingFunction : private PWLFunction {
        /// Function mapping Time spent charging to SoC. I.e. _phi(x) denotes the soc after charging
        /// for x minutes with an empty battery. Slope is of unit [kwh/time], domain is time and image is SoC
    public:
        using iterator       = PWLFunction::iterator;
        using const_iterator = PWLFunction::const_iterator;

        explicit ChargingFunction(PWLFunction::bp_container_t segments);
        explicit ChargingFunction(PWLFunction f);

        [[nodiscard]] const_iterator begin();
        [[nodiscard]] const_iterator end();
        [[nodiscard]] const_iterator begin() const;
        [[nodiscard]] const_iterator end() const;

        [[nodiscard]] double getMinimumSoC() const;
        [[nodiscard]] double getMaximumSoC() const;
        [[nodiscard]] double getFullChargeDuration() const;

        [[nodiscard]] double getSoCAfter(double time) const;
        [[nodiscard]] double getTimeRequired(double soc) const;

        [[nodiscard]] double getCharge(double from, double delta_time) const;
        [[nodiscard]] double getDuration(double q, double delta_q) const;

        [[nodiscard]] double getMaxChargingRate() const;

        friend std::ostream& operator<<(std::ostream& os, const ChargingFunction& charger);
    };

    class Charger : public std::enable_shared_from_this<Charger> {
    public:
        using charger_id = size_t;

    private:
        ChargingFunction _phi;
        charger_id _id;

    public:
        Charger(ChargingFunction phi, charger_id id);
        friend std::ostream& operator<<(std::ostream& os, const Charger& charger);
        [[nodiscard]] charger_id getID() const;
        [[nodiscard]] const ChargingFunction& getPhi() const;
        [[nodiscard]] bool operator==(const Charger&) const;
    };

    void dump(std::ostream&, Charger&);

    double charge_for_time(const ChargingFunction& phi, double from_soc, double delta_time);
    double charge_for_time(const Charger& charger, double from_soc, double delta_time);

    double duration_of_charging(const ChargingFunction& phi, double from_soc, double delta_soc);
    double duration_of_charging(const Charger& charger, double from_soc, double delta_soc);

    double initial_soc_when_charging(const ChargingFunction& phi, double final_soc, double delta_time);
    double initial_soc_when_charging(const Charger& charger, double final_soc, double delta_time);
}

#endif // FRVCP_CHARGER_HPP
