#ifndef FRVCP_BINDINGS_PWL_HPP
#define FRVCP_BINDINGS_PWL_HPP

#include <pybind11/pybind11.h>

void create_pwl_segment_bindings(pybind11::module &m);
void create_pwl_function_bindings(pybind11::module &m);

#endif // FRVCP_PWL_HPP
