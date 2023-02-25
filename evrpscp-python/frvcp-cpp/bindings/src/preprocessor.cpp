
#include "frvcp_bindings/preprocessor.hpp"
#include "frvcp/models/instance.hpp"
#include "frvcp/network.hpp"

#include <frvcp/preprocessor.hpp>

using namespace frvcp;
using namespace frvcp::models;

void create_preprocessor_bindings(pybind11::module &m) {
    auto node_binding = pybind11::class_<frvcp::Preprocessor>(m, "Preprocessor")
        .def(pybind11::init<>())
        .def("preprocess", &Preprocessor::preprocess, "Preprocess the network.");
}
