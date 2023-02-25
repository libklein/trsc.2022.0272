
#ifndef FRVCP_BINDING_CHARGER_HPP
#define FRVCP_BINDING_CHARGER_HPP

#include <pybind11/pybind11.h>

void create_charging_function_binding(pybind11::module &m);
void create_charger_binding(pybind11::module &m);

#endif // FRVCP_CHARGER_HPP
