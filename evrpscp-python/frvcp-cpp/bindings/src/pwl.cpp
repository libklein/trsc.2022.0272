#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "frvcp/cost_profile.hpp"
#include "frvcp_bindings/binding_helpers.hpp"
#include "frvcp_bindings/pwl.hpp"
#include <frvcp/models/pwl.hpp>
#include <frvcp/models/pwl_util.hpp>

using namespace frvcp::models;

void create_pwl_segment_bindings(pybind11::module& m) {
    using Breakpoint = PWLFunction::breakpoint_t;
    pybind11::class_<Breakpoint>(m, "PWLSegment")
        .def(pybind11::init<double, double>())
        .def(pybind11::init<double, double, double>())
        .def_readwrite("slope", &Breakpoint::slope)
        .def_readwrite("domain", &Breakpoint::domain)
        .def_readwrite("image", &Breakpoint::image)
        .def("__eq__", [](const Breakpoint& lhs, const Breakpoint& rhs) { return lhs == rhs; })
        .def("__repr__", &ostream_to_string<Breakpoint>);
}

namespace conversion {
    PWLFunction construct_from_breakpoints(std::vector<PWLFunction::bp_container_t::value_type> breakpoints,
        bool optimize = true, bool force_recomputation = false) {
        return construct_from_breakpoints(
            PWLFunction::bp_container_t { breakpoints.begin(), breakpoints.end() }, optimize, force_recomputation);
    }
}

void create_pwl_function_bindings(pybind11::module& m) {
    auto pwl_binding
        = pybind11::class_<PWLFunction>(m, "PWLFunction")
              .def(pybind11::init([](std::vector<PWLFunction::bp_container_t::value_type> vec) {
                  return conversion::construct_from_breakpoints(std::move(vec), false, false);
              }))
              .def("getImageLowerBound", &PWLFunction::getImageLowerBound)
              .def_property_readonly("image_lower_bound", &PWLFunction::getImageLowerBound)
              .def("getImageUpperBound", &PWLFunction::getImageUpperBound)
              .def_property_readonly("image_upper_bound", &PWLFunction::getImageUpperBound)
              .def("getUpperBound", &PWLFunction::getUpperBound)
              .def_property_readonly("domain_upper_bound", &PWLFunction::getUpperBound)
              .def("getLowerBound", &PWLFunction::getLowerBound)
              .def_property_readonly("domain_lower_bound", &PWLFunction::getLowerBound)
              .def("value", &PWLFunction::value)
              .def("inverse", &PWLFunction::inverse)
              .def("getBreakpoints", &PWLFunction::getBreakpoints, pybind11::return_value_policy::reference_internal)
              .def(
                  "__iter__", [](const PWLFunction& pwl) { return pybind11::make_iterator(pwl.begin(), pwl.end()); },
                  pybind11::keep_alive<0, 1>())
              .def("__eq__", [](const PWLFunction& lhs, const PWLFunction& rhs) { return lhs == rhs; })
              .def("__repr__", &ostream_to_string<PWLFunction>);

    pwl_binding.attr("LB_SLOPE") = pybind11::float_(PWLFunction::LB_SLOPE);

    m.def("create_constant_pwl", &create_constant_pwl);
    m.def("create_single_point_pwl", &create_single_point_pwl);

    m.def("is_concave", &is_concave<PWLFunction>, "Test function concavity");
    m.def("is_convex", &is_convex<PWLFunction>, "Test function convexity");

    m.def(
        "optimize_breakpoint_sequence",
        [](PWLFunction::bp_container_t segments) {
            optimize_breakpoint_sequence(segments);
            return segments;
        },
        "Optimize breakpoints of some function");
    m.def(
        "optimize_breakpoint_sequence",
        [](PWLFunction f) {
            optimize_breakpoint_sequence(f);
            return f;
        },
        "Optimize breakpoints of some function");

    pwl_binding.attr("LB_SLOPE") = pybind11::float_(PWLFunction::LB_SLOPE);

    m.def("create_constant_pwl", &create_constant_pwl);
    m.def("create_single_point_pwl", &create_single_point_pwl);

    m.def("is_concave", &is_concave<PWLFunction>, "Test function concavity");
    m.def("is_convex", &is_convex<PWLFunction>, "Test function convexity");

    m.def(
        "optimize_breakpoint_sequence",
        [](PWLFunction::bp_container_t segments) {
            optimize_breakpoint_sequence(segments);
            return segments;
        },
        "Optimize breakpoints of some function");
    m.def(
        "optimize_breakpoint_sequence",
        [](PWLFunction f) {
            optimize_breakpoint_sequence(f);
            return f;
        },
        "Optimize breakpoints of some function");

    m.def("construct_from_breakpoints", &conversion::construct_from_breakpoints,
        "Create a (possibly) optimized function from a set of breakpoints");

    m.def("shift_pwl_by", &shift_by<PWLFunction>, "Shift the passed function on x and y axes");
    m.def("clip_image", &clip_image<PWLFunction>, "Clips a piecewise linear function to the specified bounds");
    m.def("clip_domain", &clip_domain<PWLFunction>, "Clips a piecewise linear function to the specified bounds");
}