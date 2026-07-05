#!/usr/bin/env python3
"""scripts/play_slumbot.py

Play HU NLHE hands against Slumbot's public API
(https://slumbot.com/api) with a trained blueprint as hero.
Reports running win-rate in bb/100.

Slumbot game:
  SB = 50 chips, BB = 100 chips, starting stack = 20000 chips = 200 BB.

Slumbot action codes:
  "f"     = fold
  "c"     = check / call
  "k"     = check (synonym at preflop sometimes)
  "b<N>"  = bet / raise — N is hero's TOTAL street-invest target

We collapse off-tree bet sizes from Slumbot into our 4-action map by
ignoring exact sizing: any "b<N>" → treated as a single 'r' (raise) for
encoding. This means action translation only happens one direction
(opponent → us); our outgoing raises always use our trained 50%-pot
sizing.

Usage:
    python3 scripts/play_slumbot.py BLUEPRINT [--hands 200] [--cfr-cache PATH]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.deep_cfr.blueprint     import Blueprint
from src.deep_cfr.state_encoder import NLHEEncoder
from src.deep_cfr.action_slots  import legal_actions_to_slots
from src.games.postflop_nlhe    import PostflopNLHE


SLUMBOT_HOST = "https://slumbot.com"
SB_CHIPS     = 50
BB_CHIPS     = 100
STACK_CHIPS  = 20000
HEADERS      = {
    "Content-Type": "application/json",
    "User-Agent":   "imperfect-information-game/0.1",
}

_RANKS = "23456789TJQKA"
_SUITS = "cdhs"

_TOKEN_RE = re.compile(r"(b\d+|c|k|f|/)")


def card_str_to_int(s: str) -> int:
    return _RANKS.index(s[0].upper()) * 4 + _SUITS.index(s[1].lower())


def _post(path: str, body: dict, *, retries: int = 3, delay: float = 0.05) -> dict:
    if delay > 0:
        time.sleep(delay)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                SLUMBOT_HOST + path,
                data=json.dumps(body).encode("utf-8"),
                headers={**HEADERS, "Connection": "close"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = e
            time.sleep(min(2 ** attempt, 8))
    raise last_err if last_err else RuntimeError("post failed")


def replay_state(action_str: str, client_pos: int) -> dict:
    """Replay Slumbot's action string and return current betting state.

    Returns dict with keys:
        pot, my_street_invest, opp_street_invest, my_total_invest,
        opp_total_invest, street_idx (0=preflop..3=river),
        to_act_player (0=hero, 1=villain), terminal (bool),
        action_codes (per-our-game 0/1/2/3 list, length ≤ 8).
    """
    sb_player = 0  # whoever opens first preflop = SB.
    bb_player = 1
    # Slumbot convention: client_pos 0 = client is small blind (button in HU).
    hero_is_sb = client_pos == 0

    pot = SB_CHIPS + BB_CHIPS
    my_total  = SB_CHIPS  if hero_is_sb else BB_CHIPS
    opp_total = BB_CHIPS if hero_is_sb else SB_CHIPS

    # Per-street invest. Preflop opens with blinds.
    my_street  = my_total
    opp_street = opp_total

    streets, _ = _split_streets(action_str)
    street_idx = 0
    actor = sb_player if not streets[0] or street_idx == 0 else bb_player
    # Preflop first to act = SB (= 0). Postflop first to act = BB (= 1).

    action_codes: list[int] = []
    terminal = False

    for s_idx, segment in enumerate(streets):
        if s_idx > 0:
            # New street: reset per-street invest, postflop opener = BB.
            my_street, opp_street = 0, 0
            actor = bb_player

        for tok in segment:
            actor_is_hero = (actor == client_pos)

            # Translate to our 4-slot history code (0/check or fold, 1/call,
            # 2/raise, 3/all-in). Only used by the encoder; we just need it
            # to be consistent across replays.
            if tok == "f":
                terminal = True
                action_codes.append(0)
            elif tok == "c":
                # Call or check.
                if actor_is_hero:
                    diff = max(0, opp_street - my_street)
                    my_street  += diff
                    my_total   += diff
                    pot        += diff
                else:
                    diff = max(0, my_street - opp_street)
                    opp_street += diff
                    opp_total  += diff
                    pot        += diff
                # 0 if it's a check (diff==0), 1 if a call.
                action_codes.append(1 if (opp_street if actor_is_hero else my_street) > 0
                                     and (my_street if actor_is_hero else opp_street) > 0
                                     and tok == "c" else 0)
            elif tok == "k":
                action_codes.append(0)
            elif tok.startswith("b"):
                target = int(tok[1:])
                if actor_is_hero:
                    diff = max(0, target - my_street)
                    my_street  = target
                    my_total  += diff
                    pot       += diff
                else:
                    diff = max(0, target - opp_street)
                    opp_street = target
                    opp_total += diff
                    pot       += diff
                # Heuristic: if target ≈ remaining stack → all-in (code 3),
                # otherwise raise (code 2).
                actor_total = my_total if actor_is_hero else opp_total
                if actor_total >= STACK_CHIPS - 1:
                    action_codes.append(3)
                else:
                    action_codes.append(2)
            else:
                # Unknown token — ignore.
                pass
            actor = 1 - actor

    return {
        "pot":               pot,
        "my_street_invest":  my_street,
        "opp_street_invest": opp_street,
        "my_total_invest":   my_total,
        "opp_total_invest":  opp_total,
        "street_idx":        max(0, len(streets) - 1),
        "to_act_player":     0 if (actor == client_pos) else 1,
        "terminal":          terminal,
        "action_codes":      action_codes[-8:],
    }


def _split_streets(action_str: str) -> tuple[list[list[str]], list[int]]:
    if not action_str:
        return [[]], [0]
    tokens   = _TOKEN_RE.findall(action_str)
    streets  = [[]]
    bet_caps = [0]
    for tok in tokens:
        if tok == "/":
            streets.append([])
            bet_caps.append(0)
        else:
            streets[-1].append(tok)
            if tok.startswith("b"):
                bet_caps[-1] = max(bet_caps[-1], int(tok[1:]))
    return streets, bet_caps


# ── Blueprint query ──────────────────────────────────────────────────────────


def pick_action(blueprint, encoder, game, *,
                state: dict, hole_cards: list[int], board_cards: list[int],
                client_pos: int) -> str:
    """Choose hero's action by querying the blueprint. Returns a Slumbot-
    compatible action string ("f", "c", or "b<N>")."""
    pot         = state["pot"]
    my_street   = state["my_street_invest"]
    opp_street  = state["opp_street_invest"]
    my_total    = state["my_total_invest"]
    to_call     = max(0, opp_street - my_street)
    my_stack    = STACK_CHIPS - my_total
    history     = state["action_codes"]

    # Build a synthetic PostflopNLHE history that approximates the spot.
    # We use the hero's real cards; opponent's cards are placeholders
    # (encoder doesn't see them at hero's decision point) and the board
    # is padded with unused cards so PostflopNLHE doesn't error.
    used = set(hole_cards) | set(board_cards)
    filler = [c for c in range(52) if c not in used]
    p0 = tuple(hole_cards) if client_pos == 0 else tuple(filler[:2])
    p1 = tuple(filler[2:4]) if client_pos == 0 else tuple(hole_cards)
    used = used | set(p0) | set(p1)
    filler = [c for c in range(52) if c not in used]
    board_full = list(board_cards) + filler[: 5 - len(board_cards)]
    h = (p0, p1, tuple(board_full))

    # Replay our action codes through the game to reach the right info set.
    action_chars = {0: "k", 1: "c", 2: "r", 3: "a"}
    g = game
    for code in history:
        if g.is_terminal(h):
            break
        legal = g.legal_actions(h)
        want = action_chars[code]
        # Map "k"/"c" together — game distinguishes by legality.
        if want == "k" and "k" not in legal:
            want = "c" if "c" in legal else legal[0]
        if want == "c" and "c" not in legal:
            want = "k" if "k" in legal else legal[0]
        if want not in legal:
            # Off-tree fallback: pick first legal.
            want = legal[0]
        h = g.apply_action(h, want)

    if g.is_terminal(h):
        return "c"   # shouldn't be reached for a non-terminal Slumbot state

    player = g.current_player(h)
    sv = encoder.encode(h, player)
    actions = g.legal_actions(h)
    slots   = legal_actions_to_slots(actions, blueprint.metadata.action_size)
    raw     = np.asarray(blueprint.query_by_slots(sv, slots), dtype=np.float64)
    raw     = np.clip(raw, 0, None)
    s = raw.sum()
    probs = raw / s if s > 1e-9 else np.ones(len(actions)) / len(actions)

    idx = int(np.random.choice(len(actions), p=probs))
    chosen = actions[idx]

    if chosen == "f":
        return "f"
    if chosen in ("c", "k"):
        return "c"
    if chosen == "r":
        # Trained 50%-pot raise. Slumbot wants absolute hero-street target.
        # Raise SIZE above the call = 0.5 * (pot AFTER hero calls).
        pot_after_call = pot + to_call
        raise_size     = max(int(0.5 * pot_after_call), BB_CHIPS * 2)
        target         = opp_street + raise_size
        # Clamp to available chips.
        max_target     = my_street + my_stack
        target         = min(target, max_target)
        return f"b{int(target)}"
    if chosen == "a":
        # All-in: hero shoves remaining stack.
        return f"b{int(my_street + my_stack)}"
    return "c"


# ── Match loop ──────────────────────────────────────────────────────────────


def play_match(blueprint, encoder, game, n_hands: int, verbose: bool):
    total_chips    = 0
    hands_played   = 0
    t0             = time.time()
    errors         = 0

    for h_idx in range(n_hands):
        # Polite throttle between hands to avoid rate-limiting / connection
        # exhaustion. Without this we've seen the API stall after ~25 hands.
        if h_idx > 0:
            time.sleep(0.2)
        try:
            r = _post("/api/new_hand", {})
        except urllib.error.URLError as e:
            errors += 1
            print(f"  [hand {h_idx+1}] new_hand failed: {e}")
            time.sleep(5)
            continue
        except Exception as e:
            errors += 1
            print(f"  [hand {h_idx+1}] new_hand exception: {e}")
            time.sleep(5)
            continue

        token      = r["token"]
        client_pos = r["client_pos"]
        hole       = [card_str_to_int(c) for c in r["hole_cards"]]
        action_str = r.get("action", "")

        while True:
            if "winnings" in r:
                total_chips  += int(r["winnings"])
                hands_played += 1
                break
            board      = [card_str_to_int(c) for c in r.get("board", [])]
            state      = replay_state(action_str, client_pos)
            our_action = pick_action(
                blueprint, encoder, game,
                state=state, hole_cards=hole, board_cards=board,
                client_pos=client_pos,
            )
            try:
                r = _post("/api/act", {"token": token, "incr": our_action})
                action_str = r.get("action", action_str)
            except urllib.error.URLError as e:
                errors += 1
                print(f"  [hand {h_idx+1}] act failed: {e}")
                break
            except Exception as e:
                errors += 1
                print(f"  [hand {h_idx+1}] unexpected: {e}")
                break

        if verbose and (h_idx + 1) % 10 == 0:
            avg_mbb = (total_chips / max(hands_played, 1)
                       / BB_CHIPS * 1000.0)
            rate    = hands_played / max(time.time() - t0, 1e-9)
            print(f"  [{h_idx+1:>4d}/{n_hands}] avg={avg_mbb:+.1f} mbb/h  "
                  f"running_total={total_chips:+d} chips  "
                  f"({rate:.1f} hands/s)")

    elapsed = time.time() - t0
    if hands_played == 0:
        print("No hands completed.")
        return None
    avg_mbb = (total_chips / hands_played / BB_CHIPS * 1000.0)
    bb100   = avg_mbb / 10.0
    print()
    print(f"  Played {hands_played} hands in {elapsed/60:.1f} min "
          f"(errors={errors})")
    print(f"  Total: {total_chips:+d} chips  ({total_chips / BB_CHIPS:+.1f} BB)")
    print(f"  Avg:   {avg_mbb:+.1f} mbb/hand = {bb100:+.2f} bb/100")
    return bb100


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("blueprint",   help="Blueprint directory")
    p.add_argument("--hands",     type=int, default=200)
    p.add_argument("--cfr-cache", type=str, default="",
                   help="Same cache the blueprint trained against.")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--verbose",   "-v", action="store_true")
    args = p.parse_args()

    np.random.seed(args.seed)

    bp = Blueprint.load(args.blueprint, device="cpu")
    _rfs = (tuple(bp.metadata.raise_fractions)
            if bp.metadata.raise_fractions
            else (bp.metadata.raise_fraction,))
    if bp.metadata.starting_stack != 200.0:
        print(f"[warn] blueprint stack={bp.metadata.starting_stack}BB but "
              f"Slumbot plays 200BB. Strategy may be miscalibrated.")
    game = PostflopNLHE(
        starting_stack=bp.metadata.starting_stack,
        max_raises_per_street=bp.metadata.max_raises,
        raise_fractions=_rfs,
    )
    cache = None
    if args.cfr_cache:
        from src.deep_cfr.cfr_cache import CFRCache
        cache = CFRCache.load(args.cfr_cache)
        print(f"Loaded CFR cache: {len(cache)} entries")
    encoder = NLHEEncoder(
        starting_stack=bp.metadata.starting_stack,
        raise_fractions=_rfs,
        cfr_cache=cache,
    )

    print(f"Playing {args.hands} hands vs Slumbot (200BB)...")
    play_match(bp, encoder, game, args.hands, args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
