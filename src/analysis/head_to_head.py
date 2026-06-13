"""
Head-to-head match between two blueprints.

Two-player NLHE: alternating positions over n_hands.
Each player queries its own blueprint, samples actions stochastically.
Returns mean win-rate of blueprint A in mbb/hand (positive = A beats B).

Why this exists:
    estimate_exploitability is a one-sided diagnostic (best response vs each
    blueprint independently). Two blueprints with similar exploitability can
    still have a meaningful EV gap against each other if their abstraction /
    action mixture differs systematically. h2h surfaces that gap directly.
"""

import numpy as np
from itertools import combinations


def _showdown_evaluator():
    """Lazy import of the 7-card evaluator (avoid module-load cost when unused)."""
    from src.abstraction.equity import evaluate_7card
    return evaluate_7card


def _detect_allin_street(history, game) -> int | None:
    """Return the street_idx at which both players first became all-in, or
    None if showdown was reached with both players still having chips.

    Records the street BEFORE the action that triggered the all-in, NOT the
    street after — auto-advance bumps street_idx the instant the call lands,
    but for equity we want the cards visible at the moment of the decision.
    """
    actions = history[3:]
    base = history[:3]
    for i in range(len(actions)):
        prefix_before = base + actions[:i]
        st_before = game._parse_state(prefix_before)
        prefix_after = base + actions[:i + 1]
        st_after = game._parse_state(prefix_after)
        if st_after["stacks"][0] <= 1e-6 and st_after["stacks"][1] <= 1e-6:
            return int(st_before["street_idx"])
    return None


def _equity_at_street(hole_a: tuple, hole_b: tuple, board: tuple,
                       street_idx: int) -> float:
    """Hero (player a) equity vs villain (player b) over remaining board.

    street_idx: street at which betting ended (0=preflop, 1=flop, 2=turn, 3=river).
        Cards in `board` up to that street are 'visible'; beyond is runout.
        For preflop all-in we'd need to enumerate 5-card boards (slow);
        we use the equity_vs_specific MC.
    """
    evaluate = _showdown_evaluator()
    n_visible = (0, 3, 4, 5)[min(street_idx, 3)]
    visible = tuple(board[:n_visible])
    # Cards still to come.
    needed = 5 - n_visible
    if needed == 0:
        # River — deterministic.
        ha = evaluate(tuple(hole_a) + visible)
        hb = evaluate(tuple(hole_b) + visible)
        if ha > hb: return 1.0
        if hb > ha: return 0.0
        return 0.5
    # Build the deck of live cards (not in hands or visible board).
    used = set(hole_a) | set(hole_b) | set(visible)
    live = [c for c in range(52) if c not in used]
    if needed == 1:
        # Turn: exact enumeration over 1 card.
        wins, ties, total = 0, 0, 0
        for c in live:
            full = visible + (c,)
            ha = evaluate(tuple(hole_a) + full)
            hb = evaluate(tuple(hole_b) + full)
            if ha > hb: wins += 1
            elif ha == hb: ties += 1
            total += 1
        return (wins + 0.5 * ties) / total
    if needed == 2:
        # Flop: exact enumeration over C(remaining, 2) combos.
        wins, ties, total = 0, 0, 0
        for combo in combinations(live, 2):
            full = visible + combo
            ha = evaluate(tuple(hole_a) + full)
            hb = evaluate(tuple(hole_b) + full)
            if ha > hb: wins += 1
            elif ha == hb: ties += 1
            total += 1
        return (wins + 0.5 * ties) / total
    # Preflop all-in: 2.6M combos, too many for exact. MC with fixed seed.
    rng = np.random.default_rng((hash((tuple(sorted(hole_a)),
                                       tuple(sorted(hole_b))))) & 0x7FFFFFFF)
    n_sims = 2000
    wins = 0.0
    for _ in range(n_sims):
        runout = rng.choice(live, size=5, replace=False)
        full = tuple(int(c) for c in runout)
        ha = evaluate(tuple(hole_a) + full)
        hb = evaluate(tuple(hole_b) + full)
        if ha > hb: wins += 1.0
        elif ha == hb: wins += 0.5
    return wins / n_sims


def ev_adjusted_payoffs(history, game) -> tuple[float, float]:
    """Like game.terminal_payoffs but replaces post-all-in runout variance with
    expected value. Fold terminals are unchanged. Showdowns where both players
    had chips left at every street (no all-in) use actual outcome (deterministic
    given the deal). Showdowns reached via all-in get equity over remaining
    board cards.
    """
    if not game.is_terminal(history):
        raise ValueError("Non-terminal history passed to ev_adjusted_payoffs")
    state = game._parse_state(history)
    invested = state["invested"]

    # Fold path: deterministic, no adjustment.
    if state["folded"][0] or state["folded"][1]:
        return game.terminal_payoffs(history)

    # Showdown. Detect all-in street; if neither player went all-in, the hand
    # was played through normally (every revealed card was reactable) and the
    # actual showdown IS the EV.
    allin_street = _detect_allin_street(history, game)
    if allin_street is None or allin_street >= 3:
        return game.terminal_payoffs(history)

    # Equity-adjusted payoff over remaining board cards.
    hole_a, hole_b = history[0], history[1]
    board = history[2]
    eq_a = _equity_at_street(hole_a, hole_b, board, allin_street)
    # Pot share. Total committed = invested[0] + invested[1].
    # In zero-sum: player 0's gain = (eq_a * total_pot) − invested[0]
    pot = invested[0] + invested[1]
    p0 = eq_a * pot - invested[0]
    p1 = (1 - eq_a) * pot - invested[1]
    return (p0, p1)


def play_hand(bp_a, bp_b, game, encoder_a, encoder_b, deal, hero: int,
              rng: np.random.Generator, ev_adjusted: bool = False) -> float:
    """Play one hand. hero=0 → bp_a is SB, bp_b is BB. Returns bp_a payoff.

    encoder_a / encoder_b can differ if the blueprints were trained on
    different state-size encoders (e.g. K_BOARD=8 vs K_BOARD=16). Each player
    queries its own blueprint with its matching encoder so legacy blueprints
    keep working after architecture bumps.

    ev_adjusted=True replaces post-all-in showdown payoffs with equity over
    remaining board cards — eliminates runout variance for all-in spots.
    """
    history = deal
    while not game.is_terminal(history):
        player  = game.current_player(history)
        actions = game.legal_actions(history)
        is_hero = (player == hero)
        enc = encoder_a if is_hero else encoder_b
        bp  = bp_a      if is_hero else bp_b
        state = enc.encode(history, player)
        probs = bp.query(state, len(actions))
        probs = np.asarray(probs, dtype=np.float64)
        s = probs.sum()
        if s <= 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs /= s
        idx = int(rng.choice(len(actions), p=probs))
        history = game.apply_action(history, actions[idx])
    # Convention: bp_a is the "hero" — payoff for whichever seat A took.
    if ev_adjusted:
        return float(ev_adjusted_payoffs(history, game)[hero])
    return float(game.terminal_payoffs(history)[hero])


def match_crn(bp_a, bp_b, game, encoder_a, encoder_b=None,
              n_pairs: int = 2500, seed: int = 0,
              progress: bool = False,
              ev_adjusted: bool = False) -> dict:
    """
    Common Random Numbers (CRN) paired-comparison match.

    For each pair i in [0, n_pairs), one deal D_i is generated deterministically
    from the master seed. D_i is then played TWICE:
        Round 1: bp_a sits in hero_pos = i%2, bp_b in the other seat.
        Round 2: bp_b sits in hero_pos, bp_a in the other seat.
    Both rounds reuse the SAME action-sampling RNG seed so the only difference
    between the rounds is which blueprint occupies hero_pos — the random
    structure (cards, mixed-strategy coinflips) is shared.

    The variance of the paired difference Δ_i = payoff_a − payoff_b is much
    smaller than the variance of either payoff alone (Var(Δ) = Var(A)+Var(B)−
    2·Cov(A,B), with positive covariance from shared deal). Typically 5–10×
    fewer pairs needed than naive h2h for the same significance threshold.

    Returns a dict with mean/stderr in mbb/hand (per-pair, i.e. per matched
    deal — each "pair" consumes 2 hands).
    """
    enc_a = encoder_a
    enc_b = encoder_b if encoder_b is not None else encoder_a
    diffs = np.zeros(n_pairs, dtype=np.float64)

    has_sample = hasattr(game, "sample_deal")
    master = np.random.default_rng(seed)
    pair_seeds = master.integers(1, 2**63 - 1, size=n_pairs, dtype=np.int64)

    for i in range(n_pairs):
        pseed     = int(pair_seeds[i])
        deal_rng  = np.random.default_rng(pseed)
        if has_sample:
            deal = game.sample_deal(deal_rng)
        else:
            init = game.initial_histories()
            probs = np.array([p for _, p in init])
            deal = init[int(deal_rng.choice(len(init), p=probs))][0]

        hero_pos    = i % 2                 # alternate positional bias
        action_seed = pseed ^ 0xDEADBEEF

        # Round 1: bp_a in hero_pos
        rng1 = np.random.default_rng(action_seed)
        payoff_a = play_hand(bp_a, bp_b, game, enc_a, enc_b, deal,
                             hero=hero_pos, rng=rng1, ev_adjusted=ev_adjusted)

        # Round 2: bp_b in hero_pos — swap the role arguments. Use the SAME
        # action_seed so any random choice common to both blueprints (chance
        # nodes, identical mixed actions) is sampled identically.
        rng2 = np.random.default_rng(action_seed)
        payoff_b = play_hand(bp_b, bp_a, game, enc_b, enc_a, deal,
                             hero=hero_pos, rng=rng2, ev_adjusted=ev_adjusted)

        # Δ = bp_a's payoff in hero_pos − bp_b's payoff in hero_pos.
        diffs[i] = payoff_a - payoff_b

        if progress and (i + 1) % max(1, n_pairs // 10) == 0:
            print(f"  crn: {i+1}/{n_pairs}")

    bb = float(getattr(game, "bb", 2.0))
    starting_stack = float(getattr(game, "starting_stack", 200.0))
    mbb_per_pair = (diffs / bb) * 1000.0
    mean_diff    = float(mbb_per_pair.mean())
    # Stderr of the paired-difference estimator (each pair = one independent obs).
    stderr       = float(mbb_per_pair.std(ddof=1) / np.sqrt(n_pairs))
    return {
        "win_rate_mbb": mean_diff,
        "stderr_mbb":   stderr,
        "n_pairs":      n_pairs,
        "n_hands":      n_pairs * 2,
        "starting_stack": starting_stack,
    }


def match(bp_a, bp_b, game, encoder, encoder_b=None,
          n_hands: int = 2000, seed: int = 0,
          progress: bool = False,
          ev_adjusted: bool = False) -> dict:
    """
    Play n_hands between bp_a and bp_b. Alternates seats every hand to remove
    positional bias. Same deals are reused with swapped positions when possible
    so positional EV doesn't dominate the sample.

    Returns:
        dict with keys:
          win_rate_mbb : float — bp_a EV in mbb/hand (positive = A wins)
          stderr_mbb   : float — standard error of mean (mbb/hand)
          n_hands      : int
    """
    rng = np.random.default_rng(seed)
    payoffs = np.zeros(n_hands, dtype=np.float64)

    has_sample = hasattr(game, "sample_deal")
    enc_a = encoder
    enc_b = encoder_b if encoder_b is not None else encoder

    for i in range(n_hands):
        if has_sample:
            deal = game.sample_deal(rng)
        else:
            init = game.initial_histories()
            probs = np.array([p for _, p in init])
            deal = init[int(rng.choice(len(init), p=probs))][0]
        # Alternate seat
        hero = i % 2
        payoffs[i] = play_hand(bp_a, bp_b, game, enc_a, enc_b, deal, hero, rng,
                               ev_adjusted=ev_adjusted)
        if progress and (i + 1) % max(1, n_hands // 10) == 0:
            print(f"  h2h: {i+1}/{n_hands}")

    starting_stack = float(getattr(game, "starting_stack", 200.0))
    # mbb/hand: payoff in stack-units → mbb by * 1000 / BB. BB ~ 2 in 200BB
    # setup (sb=1, bb=2 by default). Use game's bb if available.
    bb = float(getattr(game, "bb", 2.0))
    mbb_per_hand = (payoffs / bb) * 1000.0
    win_rate    = float(mbb_per_hand.mean())
    stderr      = float(mbb_per_hand.std(ddof=1) / np.sqrt(n_hands))
    return {
        "win_rate_mbb": win_rate,
        "stderr_mbb":   stderr,
        "n_hands":      n_hands,
        "starting_stack": starting_stack,
    }
