"""src/deep_cfr/action_slots.py

Single source of truth for the Python action symbol → network output slot
mapping, mirroring the C++ ``nlhe_action_to_slot`` helper in
``src/cpp_engine/include/nlhe_game.hpp``.

Why this exists
---------------
Python PostflopNLHE.legal_actions() returns symbolic actions (``'f'``,
``'c'``, ``'k'``, ``'r'``, ``'r0'``..``'r2'``, ``'a'``) in a list. Callers
used to query the blueprint with ``bp.query(state, len(actions))`` and
zip the returned probabilities one-to-one with that list.

That zipping was **only correct when** the list of legal actions
happened to match the C++ enum order [0..n-1]. The moment a state has a
non-contiguous set of legal actions (e.g. no-bet postflop ``['c','r','a']``
maps to enum slots [0, 2, 5]) the zip becomes wrong: the network was
trained at C++ enum slots, but Python read them as contiguous slots.

Combined with the single-raise puu's ALL_IN slot 5 (which doesn't fit
into max_actions=4), this turned h2h play, LBR measurement, and
distillation targets into systematic misalignment bugs.

This module + ``Blueprint.query_by_slots`` resolve the ambiguity:
  1. ``action_symbol_to_slot(symbol, max_actions)`` returns the
     network slot for any Python action symbol.
  2. ``Blueprint.query_by_slots(state, slot_indices)`` reads exactly
     those slots, renormalised over the legal subset.
"""

from __future__ import annotations


# Network slot for each Python action symbol BEFORE the single-raise
# ALL_IN remap. Mirrors the NLHEAction enum in nlhe_game.hpp.
_BASE_SLOT = {
    "f":  0,   # FOLD_OR_CHECK (fold context)
    "c":  0,   # FOLD_OR_CHECK (check context)
    "k":  1,   # CALL
    "r":  2,   # RAISE_0 (legacy single-raise symbol)
    "r0": 2,   # RAISE_0
    "r1": 3,   # RAISE_1
    "r2": 4,   # RAISE_2
    "a":  5,   # ALL_IN (remapped to 3 in single-raise mode; see below)
}


def action_symbol_to_slot(action: str, max_actions: int) -> int:
    """Return the network output slot for ``action`` under ``max_actions``.

    The single-raise puu uses ``max_actions == 4`` and remaps the
    ALL_IN enum value (5) onto slot 3 so the all-in probability lands
    in a slot that fits the network output and gets actual training
    signal (mirrors the C++ ``nlhe_action_to_slot`` helper).

    The multi-raise puu uses ``max_actions == 6`` and keeps each enum
    value as its own slot.
    """
    if action not in _BASE_SLOT:
        raise KeyError(
            f"Unknown action symbol {action!r}. Expected one of "
            f"{sorted(_BASE_SLOT.keys())}."
        )
    slot = _BASE_SLOT[action]
    if slot == 5 and max_actions == 4:
        return 3
    return slot


def legal_actions_to_slots(actions, max_actions: int) -> list[int]:
    """Vectorised helper: map a list of Python action symbols to slots."""
    return [action_symbol_to_slot(str(a), max_actions) for a in actions]
