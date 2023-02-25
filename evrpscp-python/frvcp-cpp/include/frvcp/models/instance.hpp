
#ifndef FRVCP_INSTANCE_HPP
#define FRVCP_INSTANCE_HPP

#include "frvcp/models/battery.hpp"
#include "frvcp/models/charger.hpp"
#include "frvcp/models/tour.hpp"
#include "frvcp/label_fwd.hpp"
#include <deque>

namespace frvcp::models {

    class PWLFunction;

    class Instance {
    public:
        using charger_id = std::deque<Charger>::size_type;
        using tour_id = std::deque<Tour>::size_type;
    private:
        Battery _battery;
        std::deque<Charger> _chargers;
        std::deque<Tour> _tours;

        //double _latest_arrival_time;
        //double _max_charging_rate;

        //std::vector<double> _consumption_remaining;
        //std::vector<double> _cost_remaining;
    public:
        explicit Instance(Battery battery);

        Charger& addCharger(Charger charger);
        Tour& addTour(Tour tour);

        [[nodiscard]] const Battery& getBattery() const;
        [[nodiscard]] const Charger& getCharger(charger_id id) const;
        [[nodiscard]] const Tour& getTour(tour_id id) const;

        [[nodiscard]] std::size_t getNumberOfChargers() const;
        [[nodiscard]] std::size_t getNumberOfTours() const;

        /// Get the total energy required to serve all assigned tours
        [[nodiscard]] double getTotalConsumption() const;
    };

}

#endif // FRVCP_INSTANCE_HPP
