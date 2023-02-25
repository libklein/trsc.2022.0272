#include "frvcp_bindings/solver.hpp"
#include <frvcp/solver.hpp>

using namespace frvcp;
using namespace frvcp::models;

void create_solver_bindings(pybind11::module& m) {
    auto solver_binding = pybind11::class_<Solver>(m, "Solver")
        .def(pybind11::init<const TimeExpandedNetwork&>())
        .def("set_network", &Solver::setNetwork, "Set the network the solver operates on")
        .def("reset", &Solver::reset, "Resets the solver, freeing all labels.")
        .def("solve", &Solver::solve, "Determine the cheapest path in the configured network."
                                      "Paths more expensive than UB are eagerly discarded.", pybind11::return_value_policy::reference_internal);
#ifdef ENABLE_CALLBACKS
    solver_binding.def("on_label_created", [](Solver &solver, pybind11::function& callback){
        solver.on_label_create([callback](const Label& label){
            pybind11::bool_ discard_label = callback(label);
            return static_cast<bool>(discard_label);
        });
    });
#endif



        /*.def("solve_cb", [](Solver &solver, double ub, pybind11::function& callback){
                solver.solve_cb(ub, [&callback](const Label* label) -> bool {
                    auto schedule = util::create_charging_schedule(label);
                    pybind11::bool_ continue_solving = callback(schedule);
                    return static_cast<bool>(continue_solving);
                });
            }, "Determine the cheapest path in the configured network."
                                      "Paths more expensive than UB are eagerly discarded.");*/
}