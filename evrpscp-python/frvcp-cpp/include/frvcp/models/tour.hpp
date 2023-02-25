
#ifndef FRVCP_TOUR_HPP
#define FRVCP_TOUR_HPP

#include "frvcp/definitions.hpp"

#include <cstddef>
#include <ostream>
#include <memory>

namespace frvcp::models {
    class Tour : public std::enable_shared_from_this<Tour> {
    public:
        using tour_id = std::size_t;

    private:
        tour_id _id;
        double _consumption;
        Period _earliest_departure_period;
        Period _latest_departure_period;
        Period _duration;
    public:
        Tour(tour_id id, double consumption, Period earliest_departure_period, Period latest_departure_period,
            Period duration);
        friend std::ostream& operator<<(std::ostream& os, const Tour& tour);
        [[nodiscard]] tour_id getID() const;
        [[nodiscard]] double getConsumption() const;
        [[nodiscard]] Period getLatestDeparturePeriod() const;
        [[nodiscard]] Period getEarliestDeparturePeriod() const;
        [[nodiscard]] Period getDuration() const;

        void setID(tour_id id);
    };
}

#endif // FRVCP_TOUR_HPP
