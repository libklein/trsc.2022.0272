#include "frvcp/models/tour.hpp"

#include <cassert>

namespace frvcp::models {

    std::ostream& operator<<(std::ostream& os, const Tour& tour) {
        os << "T" << tour._id << " [" << tour.getConsumption() << "]";
        return os;
    }

    Tour::tour_id Tour::getID() const { return _id; }

    double Tour::getConsumption() const { return _consumption; }

    Tour::Tour(Tour::tour_id id, double consumption, Period earliest_departure_period, Period latest_departure_period,
        Period duration)
        : _id(id)
        , _consumption(consumption)
        , _earliest_departure_period(earliest_departure_period)
        , _latest_departure_period(latest_departure_period)
        , _duration(duration) {
        assert(getConsumption() >= 0.0);
        assert(getEarliestDeparturePeriod() <= getLatestDeparturePeriod());
        assert(getDuration() > 0);
    }

    Period Tour::getLatestDeparturePeriod() const { return _latest_departure_period; }

    Period Tour::getEarliestDeparturePeriod() const { return _earliest_departure_period; }

    Period Tour::getDuration() const { return _duration; }

    void Tour::setID(Tour::tour_id id) { _id = id; }
}
