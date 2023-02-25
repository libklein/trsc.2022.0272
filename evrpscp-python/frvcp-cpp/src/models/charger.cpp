#include "frvcp/models/charger.hpp"
#include "frvcp/models/pwl.hpp"

#include <cassert>
#include <iostream>
#include <ranges>
#include <sstream>

namespace frvcp::models {
    std::ostream& operator<<(std::ostream& os, const Charger& charger) {
        os << "C" << charger._id;
        return os;
    }

    Charger::Charger(ChargingFunction phi, Charger::charger_id id)
        : _phi(std::move(phi))
        , _id(id) { }

    Charger::charger_id Charger::getID() const { return _id; }

    const ChargingFunction& Charger::getPhi() const { return _phi; }

    double ChargingFunction::getCharge(double from, double delta_time) const {
        assert(delta_time >= 0.0);
        double time_offset = inverse(from) + delta_time;
        if (time_offset > getUpperBound())
            return getImageUpperBound();
        return value(time_offset);
    }

    double ChargingFunction::getDuration(double q, double delta_q) const { return inverse(q + delta_q) - inverse(q); }

    bool Charger::operator==(const Charger& other) const { return this->getID() == other.getID(); }

    double ChargingFunction::getMaxChargingRate() const {
        throw std::runtime_error("Not implemented!");
        return 0.0;
    }

    ChargingFunction::ChargingFunction(PWLFunction::bp_container_t segments)
        : ChargingFunction(PWLFunction(std::move(segments))) { }

    double ChargingFunction::getMinimumSoC() const { return getImageLowerBound(); }

    double ChargingFunction::getMaximumSoC() const { return getImageUpperBound(); }

    double ChargingFunction::getFullChargeDuration() const { return getUpperBound(); }

    ChargingFunction::const_iterator ChargingFunction::begin() { return getBreakpoints().begin(); }

    ChargingFunction::const_iterator ChargingFunction::end() { return getBreakpoints().end(); }

    ChargingFunction::const_iterator ChargingFunction::begin() const { return getBreakpoints().begin(); }

    ChargingFunction::const_iterator ChargingFunction::end() const { return getBreakpoints().end(); }

    ChargingFunction::ChargingFunction(PWLFunction f)
        : PWLFunction(std::move(f)) {
        if (!is_concave(*this)) {
            throw std::runtime_error("Cannot construct non concave charging function!");
        }
        if (this->getLowerBound() != 0.0) {
            throw std::runtime_error("Cannot construct charging function with non-0 offset");
        }
    }

    std::ostream& operator<<(std::ostream& os, const ChargingFunction& phi) {
        os << "Charging function: \n" << static_cast<const PWLFunction&>(phi);
        return os;
    }

    double ChargingFunction::getSoCAfter(double time) const {
        return PWLFunction::value(std::min(time, PWLFunction::getUpperBound()));
    }

    double ChargingFunction::getTimeRequired(double soc) const {
        return PWLFunction::inverse(std::min(soc, PWLFunction::getImageUpperBound()));
    }

    void dump(std::ostream& os, Charger& charger) { os << "Charger " << charger << " with " << charger.getPhi(); }

    double charge_for_time(const ChargingFunction& phi, double from_soc, double delta_time) {
        return phi.getSoCAfter(phi.getTimeRequired(from_soc) + delta_time);
    }

    double charge_for_time(const Charger& charger, double from_soc, double delta_time) {
        return charge_for_time(charger.getPhi(), from_soc, delta_time);
    }

    double duration_of_charging(const ChargingFunction& phi, double from_soc, double delta_soc) {
        return phi.getTimeRequired(from_soc + delta_soc) - phi.getTimeRequired(from_soc);
    }

    double duration_of_charging(const Charger& charger, double from_soc, double delta_soc) {
        return duration_of_charging(charger.getPhi(), from_soc, delta_soc);
    }

    double initial_soc_when_charging(const ChargingFunction& phi, double final_soc, double delta_time) {
        return phi.getSoCAfter(phi.getTimeRequired(final_soc) - delta_time);
    }

    double initial_soc_when_charging(const Charger& charger, double final_soc, double delta_time) {
        return initial_soc_when_charging(charger.getPhi(), final_soc, delta_time);
    }
}
