#include <frvcp/models/pwl.hpp>
#include <frvcp/util/algorithms.hpp>
#include <frvcp/util/floats.hpp>
#include <pybind11/pybind11.h>

#include "frvcp_bindings/util.hpp"

void create_utility_bindings(pybind11::module& m) {
    m.attr("EPS") = pybind11::float_(EPS);
    m.def("approx_eq", &frvcp::approx_eq);
    m.def("approx_lt", &frvcp::approx_lt);
    m.def("approx_gt", &frvcp::approx_gt);
    m.def("approx_eq", [](double x, double y) { return frvcp::approx_eq(x, y); });
    m.def("approx_lt", [](double x, double y) { return frvcp::approx_lt(x, y); });
    m.def("approx_gt", [](double x, double y) { return frvcp::approx_gt(x, y); });
    m.def("certainly_lt", &frvcp::certainly_lt);
    m.def("certainly_gt", &frvcp::certainly_gt);
    m.def("certainly_lt", [](double x, double y) { return frvcp::certainly_lt(x, y); });
    m.def("certainly_gt", [](double x, double y) { return frvcp::certainly_gt(x, y); });
}