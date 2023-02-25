
#ifndef FRVCP_BINDINGS_BATTERY_HPP
#define FRVCP_BINDINGS_BATTERY_HPP

#include <pybind11/pybind11.h>

void create_wdf_bindings(pybind11::module&);
void create_battery_bindings(pybind11::module&);

#endif // FRVCP_BATTERY_HPP
