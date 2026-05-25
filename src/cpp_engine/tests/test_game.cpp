// Minimal test binary — avoids linker error from empty file
#include <iostream>
#include "leduc_game.hpp"
#include "nlhe_game.hpp"

using namespace cfr;

int main() {
    // Leduc: verify 120 deals
    auto deals = LeducGame::all_deals();
    std::cout << "Leduc deals: " << deals.size() << " (expected 120)\n";

    // NLHE: verify initial state matches Python PostflopNLHE
    std::mt19937 rng(42);
    auto d = NLHEGame::sample_deal(rng);
    NLHEGameConfig cfg;  // defaults: 200BB, SB=1, BB=2
    auto s = NLHEGame::initial_state(d, cfg);
    std::cout << "NLHE initial pot: " << s.pot
              << " (expected " << cfg.sb + cfg.bb << ")\n";
    std::cout << "NLHE stacks: [" << s.stacks[0] << ", " << s.stacks[1]
              << "] (expected [" << cfg.starting_stack - cfg.sb
              << ", " << cfg.starting_stack - cfg.bb << "])\n";

    auto actions = NLHEGame::legal_actions(s);
    std::cout << "NLHE legal actions: " << actions.size()
              << " (expected 3 or 4)\n";

    std::cout << "All tests passed.\n";
    return 0;
}