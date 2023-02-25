#include "frvcp/models/instance.hpp"
#include "frvcp/definitions.hpp"
#include <cassert>
#include <frvcp/models/charger.hpp>
#include <frvcp/models/pwl.hpp>
#include <frvcp/models/tour.hpp>
#include <numeric>

frvcp::models::Charger& frvcp::models::Instance::addCharger(frvcp::models::Charger charger) {
    assert(std::none_of(
        _chargers.begin(), _chargers.end(), [&charger](const auto& ch) { return ch.getID() == charger.getID(); }));
    assert(charger.getPhi().getMinimumSoC() == getBattery().getMinimumSoC());
    assert(charger.getPhi().getMaximumSoC() == getBattery().getMaximumSoC());

    _chargers.push_back(std::move(charger));
    return _chargers.back();
}

auto frvcp::models::Instance::addTour(Tour tour) -> Tour& {
    assert(std::none_of(_tours.begin(), _tours.end(), [&tour](const Tour& t) { return t.getID() == tour.getID(); }));
    _tours.push_back(tour);
    return _tours.back();
}

frvcp::models::Instance::Instance(frvcp::models::Battery battery)
    : _battery(std::move(battery)) { }

const frvcp::models::Charger& frvcp::models::Instance::getCharger(charger_id id) const { return _chargers[id]; }

const frvcp::models::Tour& frvcp::models::Instance::getTour(tour_id id) const { return _tours[id]; }

std::size_t frvcp::models::Instance::getNumberOfChargers() const { return _chargers.size(); }

std::size_t frvcp::models::Instance::getNumberOfTours() const { return _tours.size(); }

double frvcp::models::Instance::getTotalConsumption() const {
    return std::accumulate(
        _tours.begin(), _tours.end(), 0.0, [](double sum, const Tour& t) { return sum + t.getConsumption(); });
}

const frvcp::models::Battery& frvcp::models::Instance::getBattery() const { return _battery; }
/*
double frvcp::models::Instance::getLatestArrivalTime() const {
    return _latest_arrival_time;
}

double frvcp::models::Instance::getMaxChargingRate() const {
    return _max_charging_rate;
}

double frvcp::models::Instance::getRemainingEnergyConsumption(const frvcp::tour_set_t& traveled_tours) const {
    return _consumption_remaining[traveled_tours.to_ulong()];
}
double frvcp::models::Instance::getRemainingTravelCost(const frvcp::tour_set_t& traveled_tours) const {
    return _cost_remaining[traveled_tours.to_ulong()];
}*/
