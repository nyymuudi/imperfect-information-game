#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include "leduc_game.hpp"
#include "mccfr.hpp"
#include "nlhe_game.hpp"
#include "nlhe_mccfr.hpp"
#include "torch_model.hpp"

namespace py = pybind11;
using namespace cfr;

PYBIND11_MODULE(cfr_engine, m) {
    m.doc() = "C++ MCCFR engine — Leduc + NLHE + LibTorch (Deep CFR backend)";

    // ════════════════════════════════════════════════════════════════════════
    // LEDUC
    // ════════════════════════════════════════════════════════════════════════
    py::enum_<Action>(m, "Action")
        .value("FOLD",  FOLD).value("CHECK", CHECK)
        .value("CALL",  CALL).value("RAISE", RAISE)
        .export_values();

    py::class_<LeducDeal>(m, "LeducDeal")
        .def_readonly("private_cards", &LeducDeal::private_cards)
        .def_readonly("community_card",&LeducDeal::community_card)
        .def_readonly("probability",   &LeducDeal::probability);

    py::class_<LeducState>(m, "LeducState")
        .def_readonly("current_player",    &LeducState::current_player)
        .def_readonly("round",             &LeducState::round)
        .def_readonly("pot",               &LeducState::pot)
        .def_readonly("terminal",          &LeducState::terminal)
        .def_readonly("payoff_p0",         &LeducState::payoff_p0)
        .def_readonly("raises_this_round", &LeducState::raises_this_round)
        .def("private_card",  [](const LeducState& s, int p){ return (int)s.private_cards[p]; })
        .def("contribution",  [](const LeducState& s, int p){ return s.contributions[p]; })
        .def("folded",        [](const LeducState& s, int p){ return s.folded[p]; });

    py::class_<LeducGame>(m, "LeducGame")
        .def(py::init<>())
        .def_static("all_deals",     &LeducGame::all_deals)
        .def_static("initial_state", &LeducGame::initial_state)
        .def_static("legal_actions", &LeducGame::legal_actions)
        .def_static("apply_action",  &LeducGame::apply_action)
        .def_static("info_set_key",  &LeducGame::info_set_key)
        .def_static("hand_strength", &LeducGame::hand_strength)
        .def_static("bet_size",      &LeducGame::bet_size);

    py::class_<TraversalConfig>(m, "TraversalConfig")
        .def(py::init<>())
        .def_readwrite("n_traversals",      &TraversalConfig::n_traversals)
        .def_readwrite("iteration",         &TraversalConfig::iteration)
        .def_readwrite("regret_capacity",   &TraversalConfig::regret_capacity)
        .def_readwrite("strategy_capacity", &TraversalConfig::strategy_capacity)
        .def_readwrite("collect_strategy",  &TraversalConfig::collect_strategy)
        .def_readwrite("seed",              &TraversalConfig::seed);

    py::class_<MCCFREngine::BufferExport>(m, "BufferExport")
        .def_readonly("info_sets",  &MCCFREngine::BufferExport::info_sets)
        .def_readonly("actions",    &MCCFREngine::BufferExport::actions)
        .def_readonly("values",     &MCCFREngine::BufferExport::values)
        .def_readonly("iterations", &MCCFREngine::BufferExport::iterations)
        .def("__len__", [](const MCCFREngine::BufferExport& e){ return e.info_sets.size(); });

    py::class_<MCCFREngine>(m, "MCCFREngine")
        .def(py::init<const TraversalConfig&>())
        .def("run_traversals",         &MCCFREngine::run_traversals,
             py::arg("traversing_player"), py::arg("strategy_fn"),
             py::call_guard<py::gil_scoped_release>())
        .def("run_traversals_uniform", &MCCFREngine::run_traversals_uniform,
             py::arg("traversing_player"),
             py::call_guard<py::gil_scoped_release>())
        .def("clear_buffers",          &MCCFREngine::clear_buffers)
        .def("set_iteration",          &MCCFREngine::set_iteration)
        .def("export_regret_buffer",   &MCCFREngine::export_regret_buffer)
        .def("export_strategy_buffer", &MCCFREngine::export_strategy_buffer)
        .def("regret_buffer_size",  [](const MCCFREngine& e){ return e.regret_buffer().size(); })
        .def("strategy_buffer_size",[](const MCCFREngine& e){ return e.strategy_buffer().size(); });

    m.def("all_deals",    &LeducGame::all_deals);
    m.def("info_set_key", &LeducGame::info_set_key);
    m.attr("NUM_CARDS")  = NUM_CARDS;
    m.attr("NUM_RANKS")  = NUM_RANKS;
    m.attr("MAX_RAISES") = MAX_RAISES;
    m.attr("ANTE")       = ANTE;
    m.attr("BET_ROUND1") = BET_ROUND1;
    m.attr("BET_ROUND2") = BET_ROUND2;

    // ════════════════════════════════════════════════════════════════════════
    // NLHE
    // ════════════════════════════════════════════════════════════════════════
    py::enum_<NLHEAction>(m, "NLHEAction")
        .value("NLHE_FOLD",     NLHE_FOLD)
        .value("NLHE_CHECK",    NLHE_CHECK)
        .value("NLHE_CALL",     NLHE_CALL)
        .value("NLHE_BET_HALF", NLHE_BET_HALF)
        .value("NLHE_BET_POT",  NLHE_BET_POT)
        .value("NLHE_ALL_IN",   NLHE_ALL_IN)
        .export_values();

    py::class_<NLHETraversalConfig>(m, "NLHETraversalConfig")
        .def(py::init<>())
        .def_readwrite("n_traversals",      &NLHETraversalConfig::n_traversals)
        .def_readwrite("iteration",         &NLHETraversalConfig::iteration)
        .def_readwrite("regret_capacity",   &NLHETraversalConfig::regret_capacity)
        .def_readwrite("strategy_capacity", &NLHETraversalConfig::strategy_capacity)
        .def_readwrite("collect_strategy",  &NLHETraversalConfig::collect_strategy)
        .def_readwrite("seed",              &NLHETraversalConfig::seed)
        .def_readwrite("max_actions",       &NLHETraversalConfig::max_actions);

    py::class_<NLHEMCCFREngine>(m, "NLHEMCCFREngine")
        .def(py::init<const NLHETraversalConfig&>())
        // Callback-based (Python strategy fn)
        .def("run_traversals",
             &NLHEMCCFREngine::run_traversals,
             py::arg("traversing_player"), py::arg("strategy_fn"),
             py::call_guard<py::gil_scoped_release>())
        .def("run_traversals_uniform",
             &NLHEMCCFREngine::run_traversals_uniform,
             py::arg("traversing_player"),
             py::call_guard<py::gil_scoped_release>())
        // LibTorch-based (zero Python callbacks)
        .def("load_model",
             &NLHEMCCFREngine::load_model,
             py::arg("path"),
             "Load TorchScript regret network. Enables run_traversals_model().")
        .def("model_loaded",     &NLHEMCCFREngine::model_loaded)
        .def("run_traversals_model",
             &NLHEMCCFREngine::run_traversals_model,
             py::arg("traversing_player"),
             py::call_guard<py::gil_scoped_release>(),
             "Run traversals using LibTorch model — no Python callbacks, no GIL.")
        // Buffer management
        .def("clear_buffers",          &NLHEMCCFREngine::clear_buffers)
        .def("set_iteration",          &NLHEMCCFREngine::set_iteration)
        .def("export_regret_buffer",   &NLHEMCCFREngine::export_regret_buffer)
        .def("export_strategy_buffer", &NLHEMCCFREngine::export_strategy_buffer)
        .def("regret_buffer_size",     &NLHEMCCFREngine::regret_buffer_size)
        .def("strategy_buffer_size",   &NLHEMCCFREngine::strategy_buffer_size)
        // Strategy evaluation
        .def("load_strategy_model",    &NLHEMCCFREngine::load_strategy_model)
        .def("strategy_model_loaded",  &NLHEMCCFREngine::strategy_model_loaded)
        .def("query_preflop_strategy", &NLHEMCCFREngine::query_preflop_strategy,
             py::arg("hole0"), py::arg("hole1"))
        .def("query_strategy",         &NLHEMCCFREngine::query_strategy,
             py::arg("hole0"), py::arg("hole1"), py::arg("street"),
             py::arg("board"), py::arg("pot"), py::arg("to_call"),
             py::arg("my_stack"));

    m.attr("NLHE_STACK")         = NLHE_STACK;
    m.attr("NLHE_BB")            = NLHE_BB;
    m.attr("NLHE_DECK_SIZE")     = NLHE_DECK_SIZE;
    m.attr("NLHE_STATE_SIZE")    = NLHEStateEncoder::STATE_SIZE;
    m.attr("TORCH_AVAILABLE")    = bool(
#ifdef CFR_TORCH_AVAILABLE
        true
#else
        false
#endif
    );
}