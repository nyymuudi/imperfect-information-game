"""
State encoder for Deep CFR.

Converts game histories into fixed-size numeric vectors suitable
for neural network input. Each game requires a specific encoder
because the state representation depends on the game structure.

Design principle: the encoder must use ONLY information available
to the current player (info set), never the full game state.
This prevents abstraction leakage.
"""

import numpy as np
from abc import ABC, abstractmethod
from ..games.base import History


class StateEncoder(ABC):
    """Abstract state encoder interface."""

    @abstractmethod
    def state_size(self) -> int:
        """Return the fixed size of the output state vector."""
        ...

    @abstractmethod
    def encode(self, history: History, player: int) -> np.ndarray:
        """
        Encode a history from a player's perspective.

        MUST use only information visible to the player:
        - Their own private cards
        - Public information (community cards, action sequence)
        - NOT the opponent's cards

        Returns a 1D numpy array of size state_size().
        """
        ...


class LeducEncoder(StateEncoder):
    """
    Leduc Hold'em state encoder.

    State vector (20 dimensions):
        [0:3]   Private card rank (one-hot: J=0, Q=1, K=2)
        [3:6]   Community card rank (one-hot, zeros if not revealed)
        [6]     Community revealed (0 or 1)
        [7]     Pot size (normalized by max pot = 26)
        [8]     Amount to call (normalized by max pot)
        [9]     Current round (0 or 1)
        [10:20] Action history (last 10 actions, encoded as:
                check=0.0, raise=0.5, fold=-1.0, call=1.0, empty=0.0)
    """

    RANK_IDX = {"J": 0, "Q": 1, "K": 2}
    ACTION_ENC = {"c": 0.0, "r": 0.5, "f": -1.0, "k": 1.0}
    MAX_POT = 26.0
    HISTORY_LEN = 10

    def state_size(self) -> int:
        return 20

    def encode(self, history: History, player: int) -> np.ndarray:
        state = np.zeros(self.state_size(), dtype=np.float32)
        private_rank = history[player][0]
        state[self.RANK_IDX[private_rank]] = 1.0
        actions = history[3:]
        r1_done = self._is_round1_done(actions)
        if r1_done:
            comm_rank = history[2][0]
            state[3 + self.RANK_IDX[comm_rank]] = 1.0
            state[6] = 1.0
        pot, to_call = self._compute_pot_info(actions, player)
        state[7] = pot / self.MAX_POT
        state[8] = to_call / self.MAX_POT
        state[9] = 1.0 if r1_done else 0.0
        for i, a in enumerate(actions[-self.HISTORY_LEN:]):
            state[10 + i] = self.ACTION_ENC.get(a, 0.0)
        return state

    def _is_round1_done(self, actions: tuple) -> bool:
        if len(actions) < 2:
            return False
        for i, a in enumerate(actions):
            if a == 'f':
                return False
            if a == 'k':
                return True
            if a == 'c' and i >= 1:
                if actions[i-1] == 'c' or (i == 1 and actions[0] == 'c'):
                    return True
        return False

    def _compute_pot_info(self, actions: tuple, player: int) -> tuple[float, float]:
        pot = [1.0, 1.0]
        r1_done = False
        current = 0
        raise_sizes = {False: 2.0, True: 4.0}
        for a in actions:
            if a == 'f':
                break
            elif a == 'r':
                opp = 1 - current
                pot[current] = pot[opp] + raise_sizes[r1_done]
            elif a == 'k':
                pot[current] = pot[1 - current]
                if not r1_done:
                    r1_done = True
                    current = 0
                    continue
            current = 1 - current
        to_call = max(0, pot[1 - player] - pot[player])
        return sum(pot), to_call


class NLHEEncoder(StateEncoder):
    """
    No-Limit Hold'em state encoder for Deep CFR — card-abstracted.

    State vector (36 dimensions):
        [0:8]   Preflop hand bucket (one-hot, K=8 equal-width bins on equity)
        [8:16]  Board EHS bucket (one-hot, K=8; all zeros preflop)
        [16:20] Street (one-hot: preflop/flop/turn/river)
        [20]    Pot size (normalized by 2× starting stack)
        [21]    Amount to call (normalized)
        [22]    Own stack remaining (normalized)
        [23]    Opponent stack remaining (normalized)
        [24:32] Action history (last 8 actions encoded)
        [32]    Preflop equity of own hand (continuous, in [0,1])
        [33]    Board strength (evaluate_7card / MAX_HAND_SCORE, in [0,1])
        [34]    Pot odds = to_call / (pot + to_call)
        [35]    SPR = min(stacks) / pot (capped at 10, normalised)

    Card abstraction:
        Multiple concrete hands map to the same bucket → the same state
        vector → the buffer accumulates dense samples per abstract state
        rather than sparse samples spread across 1326 unique hand pairs.
        Preflop bucket: min(int(equity * K), K-1), K=8.
        Board bucket:   min(int(board_strength * K), K-1), K=8.
        Dims 32-33 retain the continuous signal within the bucket so the
        network can interpolate between adjacent buckets.

    Betting/street: read from PostflopNLHE._parse_state (mirrors C++ NLHEGame).
    Quantisation: dims 20-35 rounded to 1e-6 grid for stable grouping.
    """

    ACTION_ENC = {"f": -1.0, "c": 0.0, "k": 0.25, "r": 0.5, "b": 0.75, "a": 1.0}
    HISTORY_LEN = 8
    # Quantisaatioruudukko jatkuville dim 20-35:lle. Aiempi 1e-6 sai ~95%
    # tiloista uniikeiksi → card-abstraktion bucketit eivät tiivistäneet
    # bufferia tehokkaasti. 0.05 antaa ~20-tasoisen ruudukon → useampi
    # näyte per abstrakti tila. C++:n NLHEStateEncoder käyttää samaa vakiota
    # parityn säilyttämiseksi.
    FEATURE_QUANT = 0.05
    K_PREFLOP = 8
    K_BOARD   = 8

    # Equity range for normalised bucket computation.
    # Derived from the 169-hand preflop equity table (2000 sims): weakest hand
    # (72o ≈ 0.316) to strongest (AA ≈ 0.842). Equal-width bins on [0, 1] leave
    # buckets 0-1 and 7 empty; normalising to the actual range fills all K bins
    # and separates AA/72o by K-1 = 7 buckets instead of 4.
    # These constants are recomputed from the loaded table at init; the fallback
    # values here match the 2000-sim table and are used if the table is absent.
    EQ_MIN_DEFAULT: float = 0.316
    EQ_MAX_DEFAULT: float = 0.842

    _shared_equity_cache = None
    _shared_eq_min: float | None = None
    _shared_eq_max: float | None = None

    def __init__(self, starting_stack: float = 200.0, equity_sims: int = 2000,
                 K_BOARD: int | None = None,
                 bucket_scheme: str = "flat"):
        """
        bucket_scheme:
            "flat" — K_BOARD-way one-hot at [8:8+K_BOARD]. K_BOARD in (8, 16).
            "tree" — 4-way super-category one-hot at [8:12] AND 4-way
                     fine-within-super one-hot at [12:16]. Two-hot encoding,
                     K_BOARD must be 8. Gives 4×4=16 effective combinations in
                     8 dims, same state_size as flat K=8.
        """
        self.starting_stack = starting_stack
        self.norm = 2 * starting_stack
        self._equity_sims = equity_sims
        self._game_cache = None
        self.bucket_scheme = bucket_scheme
        # Allow per-instance K_BOARD override so blueprints trained on K=8
        # (state_size=36) can be queried side-by-side with K=16 blueprints
        # (state_size=44) in head-to-head matches. None = use class default.
        if K_BOARD is not None:
            if K_BOARD not in (8, 16):
                raise ValueError(f"K_BOARD must be 8 or 16, got {K_BOARD}")
            self.K_BOARD = K_BOARD
        if bucket_scheme not in ("flat", "tree", "super", "tree42"):
            raise ValueError(f"bucket_scheme must be 'flat'|'tree'|'super'|'tree42', got {bucket_scheme!r}")
        if bucket_scheme in ("tree", "super", "tree42") and self.K_BOARD != 8:
            raise ValueError(f"{bucket_scheme} scheme requires K_BOARD=8")
        if NLHEEncoder._shared_equity_cache is None:
            NLHEEncoder._shared_equity_cache = {}
            self._build_equity_cache()
        self._equity_cache = NLHEEncoder._shared_equity_cache
        self._eq_min = NLHEEncoder._shared_eq_min or self.EQ_MIN_DEFAULT
        self._eq_max = NLHEEncoder._shared_eq_max or self.EQ_MAX_DEFAULT

    def _game(self):
        """Lazily build a PostflopNLHE matching this encoder's stack, so the
        encoder reads pot/to_call/street/stacks from the SAME (corrected) state
        machine the C++ engine uses."""
        if self._game_cache is None:
            from ..games.postflop_nlhe import PostflopNLHE
            self._game_cache = PostflopNLHE(
                starting_stack=self.starting_stack,
                max_raises_per_street=2,
                raise_fractions=(0.75,),
            )
        return self._game_cache

    def _build_equity_cache(self):
        from ..abstraction.equity import preflop_equity_table
        table = preflop_equity_table(num_simulations=self._equity_sims)
        NLHEEncoder._shared_equity_cache.update(table)
        vals = list(table.values())
        if vals:
            NLHEEncoder._shared_eq_min = float(min(vals))
            NLHEEncoder._shared_eq_max = float(max(vals))

    def _get_preflop_equity(self, card1: int, card2: int) -> float:
        from ..abstraction.equity import canonical_hand_class
        hc = canonical_hand_class(card1, card2)
        return self._equity_cache.get(hc, 0.5)

    # MC EHS-cache jaettu kaikkien encoderi-instanssien välillä.
    def _get_board_strength(self, hole: tuple, board: tuple) -> float:
        """Production board strength — monotone hand-rank proxy [0, 1].

        Käyttää evaluate_7card / MAX_HAND_SCORE:n yksinkertaista normalisointia.
        Ei ole aitoa equity vaan käsiluokka-rank, mutta v8-ablation (MC EHS)
        regressoi v3:a +629 mbb/h h2h:ssa (z=4.17) — selvisi että verkko nojaa
        enemmän bucket-featureihin kuin tähän jatkuvaan signaaliin, ja MC EHS:n
        kohina vahingoittaa enemmän kuin tarkempi equity hyödyttää.
        """
        if not board:
            return 0.0
        from ..abstraction.equity import (
            evaluate_7card, evaluate_5card, MAX_HAND_SCORE,
        )
        all_cards = hole + board
        if len(all_cards) >= 5:
            val = (evaluate_7card(all_cards[:7]) if len(all_cards) >= 7
                   else evaluate_5card(all_cards[:5]))
            return min(val / MAX_HAND_SCORE, 1.0)
        return 0.0

    # PARITY: board-bucket — kategoria + subkategoria, K_BOARD=16.
    # Käsikategoria (0=high card .. 8=SF) saadaan bit-identtisesti molemmilla
    # puolilla (Python: score // 15^5, C++: score >> 24). Subkategoria lasketaan
    # suoraan (hole+board)-korttien rank-jakautumasta, ei score-pakkauksesta,
    # joten parity säilyy. HC ja pair saavat eniten sub-binnä koska ne ovat
    # selvästi yleisimpiä postflop-tilanteita.
    #
    # bucket | luokka                      | jakoperiaate
    #   0    | HC,    top rank ≤ 7 (2-9)   |
    #   1    | HC,    top rank 8-10 (T-Q)  |
    #   2    | HC,    top rank 11 (K)      |
    #   3    | HC,    top rank 12 (A)      |
    #   4    | pair,  pair rank ≤ 6 (2-8)  |
    #   5    | pair,  pair rank 7-9 (9-J)  |
    #   6    | pair,  pair rank 10-11(Q-K) |
    #   7    | pair,  pair rank 12 (A)     |
    #   8    | two pair, top pair ≤ 9      |
    #   9    | two pair, top pair ≥ 10     |
    #  10    | trips, trip rank ≤ 9        |
    #  11    | trips, trip rank ≥ 10       |
    #  12    | straight                    |
    #  13    | flush                       |
    #  14    | full house                  |
    #  15    | quads + straight flush      |
    _PACK_BASE_POW_5 = 15 ** 5

    def _board_category(self, hole: tuple, board: tuple) -> int:
        if not board:
            return 0
        from ..abstraction.equity import evaluate_7card, evaluate_5card
        all_cards = hole + board
        if len(all_cards) < 5:
            return 0
        val = (evaluate_7card(all_cards[:7]) if len(all_cards) >= 7
               else evaluate_5card(all_cards[:5]))
        return val // self._PACK_BASE_POW_5

    def _board_bucket(self, hole: tuple, board: tuple) -> int:
        """Dispatch to K_BOARD-specific bucket function (flat scheme only)."""
        if self.K_BOARD == 8:
            return self._board_bucket_k8(hole, board)
        return self._board_bucket_k16(hole, board)

    def _board_bucket_tree(self, hole: tuple, board: tuple) -> tuple[int, int]:
        """
        Return (super_idx, fine_idx) ∈ [0,4)² for the hierarchical tree scheme.

        Super-categories (4):
            0 = High card
            1 = One pair
            2 = Made non-paired   (two-pair, trips, straight, flush)
            3 = Premium           (full house, quads, straight flush)

        Fine sub-bin within each super (4 per super):
            super 0 (HC):  top rank ∈ {≤7, 8-T, K only, A only} → 0..3
                           rank 11 (K) lands in 2; ranks 8-10 (T-Q minus K) in 1.
            super 1 (Pair): pair rank ∈ {≤8, 9-J, Q-K, A} → 0..3
            super 2 (Made non-paired): category {two-pair, trips, straight, flush} → 0..3
            super 3 (Premium): {FH, Quads, SF non-royal, SF royal} → 0..3
        """
        cat = self._board_category(hole, board)
        cards = tuple(hole) + tuple(board)
        ranks = [c // 4 for c in cards] if cards else []

        if cat == 0:
            top = max(ranks) if ranks else 0
            if top == 12:   fine = 3   # A
            elif top == 11: fine = 2   # K
            elif top >= 8:  fine = 1   # T-Q
            else:           fine = 0   # 2-9
            return (0, fine)

        if cat == 1:
            counts: dict[int, int] = {}
            for r in ranks:
                counts[r] = counts.get(r, 0) + 1
            pair_ranks = [r for r, n in counts.items() if n >= 2]
            pair_rank = max(pair_ranks) if pair_ranks else 0
            if pair_rank == 12:   fine = 3
            elif pair_rank >= 10: fine = 2
            elif pair_rank >= 7:  fine = 1
            else:                 fine = 0
            return (1, fine)

        if cat in (2, 3, 4, 5):
            fine = {2: 0, 3: 1, 4: 2, 5: 3}[cat]
            return (2, fine)

        # cat 6, 7, 8 (Premium)
        if cat == 6: fine = 0      # full house
        elif cat == 7: fine = 1    # quads
        else:
            # SF: distinguish royal (top straight, A-high) vs non-royal.
            top = max(ranks) if ranks else 0
            fine = 3 if top == 12 else 2
        return (3, fine)

    def _board_bucket_k8(self, hole: tuple, board: tuple) -> int:
        """Legacy 8-way bucket used by K_BOARD=8 blueprints (e.g. v3_coarse).
        Kept for head-to-head matches against blueprints trained on K=8."""
        cat = self._board_category(hole, board)
        if cat >= 6:
            return 7   # full house, quads, SF
        if cat in (4, 5):
            return 6   # straight or flush
        if cat == 3:
            return 5   # trips
        if cat == 2:
            return 4   # two pair
        cards = tuple(hole) + tuple(board)
        ranks = [c // 4 for c in cards]
        if cat == 1:
            counts = {}
            for r in ranks:
                counts[r] = counts.get(r, 0) + 1
            pair_ranks = [r for r, n in counts.items() if n >= 2]
            pair_rank = max(pair_ranks) if pair_ranks else 0
            return 3 if pair_rank >= 7 else 2
        top_rank = max(ranks) if ranks else 0
        return 1 if top_rank >= 9 else 0

    def _board_bucket_k16(self, hole: tuple, board: tuple) -> int:
        """Compute the 16-way board bucket (category + 4/4/2/2/1/1/1/1 split)."""
        cat = self._board_category(hole, board)
        cards = tuple(hole) + tuple(board)
        ranks = [c // 4 for c in cards] if cards else []

        if cat >= 7:
            return 15  # quads + SF
        if cat == 6:
            return 14  # full house
        if cat == 5:
            return 13  # flush
        if cat == 4:
            return 12  # straight
        if cat == 3:
            # Trips: split by trip rank (rank appearing ≥ 3 times).
            counts = {}
            for r in ranks:
                counts[r] = counts.get(r, 0) + 1
            trip_ranks = [r for r, n in counts.items() if n >= 3]
            trip_rank = max(trip_ranks) if trip_ranks else 0
            return 11 if trip_rank >= 10 else 10  # rank 10 = Q
        if cat == 2:
            # Two pair: split by TOP pair rank (highest of the paired ranks).
            counts = {}
            for r in ranks:
                counts[r] = counts.get(r, 0) + 1
            pair_ranks = [r for r, n in counts.items() if n >= 2]
            top_pair = max(pair_ranks) if pair_ranks else 0
            return 9 if top_pair >= 10 else 8     # rank 10 = Q
        if cat == 1:
            # Pair: 4-way split by pair rank.
            counts = {}
            for r in ranks:
                counts[r] = counts.get(r, 0) + 1
            pair_ranks = [r for r, n in counts.items() if n >= 2]
            pair_rank = max(pair_ranks) if pair_ranks else 0
            if pair_rank == 12: return 7         # A
            if pair_rank >= 10: return 6         # Q-K
            if pair_rank >= 7:  return 5         # 9-J
            return 4                              # 2-8
        # cat == 0: high card — 4-way split by top rank.
        top_rank = max(ranks) if ranks else 0
        if top_rank == 12: return 3              # A
        if top_rank == 11: return 2              # K
        if top_rank >= 8:  return 1              # T-Q
        return 0                                  # 2-9

    def state_size(self) -> int:
        # 8 (preflop bucket) + K_BOARD + 4 (street) + 4 (betting) + 8 (history)
        # + 4 (continuous tail) = 28 + K_BOARD.
        return 28 + self.K_BOARD

    def encode(self, history: History, player: int) -> np.ndarray:
        state = np.zeros(self.state_size(), dtype=np.float32)

        my_cards = history[0] if player == 0 else history[1]

        # Preflop equity + bucket [0:8] — normalised to actual equity range
        equity = self._get_preflop_equity(my_cards[0], my_cards[1])
        eq_range = self._eq_max - self._eq_min
        if eq_range > 0:
            eq_norm = (equity - self._eq_min) / eq_range
        else:
            eq_norm = equity
        pf_bucket = min(int(eq_norm * self.K_PREFLOP), self.K_PREFLOP - 1)
        pf_bucket = max(0, pf_bucket)  # guard against equity < eq_min
        state[pf_bucket] = 1.0

        # Betting/street state from corrected state machine
        st = self._game()._parse_state(history)
        street_idx = st["street_idx"]
        n_visible = [0, 3, 4, 5][min(street_idx, 3)]

        # Dim offsets parameterised on K_BOARD so the encoder generalises 8↔16.
        BRD_OFF    = self.K_PREFLOP                       # 8
        STREET_OFF = BRD_OFF + self.K_BOARD               # 8+16=24
        BET_OFF    = STREET_OFF + 4                       # 28
        HIST_OFF   = BET_OFF + 4                          # 32
        CONT_OFF   = HIST_OFF + self.HISTORY_LEN          # 40

        # Board bucket [BRD_OFF : BRD_OFF + K_BOARD]
        visible_board = history[2][:n_visible] if n_visible > 0 else ()
        board_str = self._get_board_strength(my_cards, visible_board)
        if n_visible >= 3:
            if self.bucket_scheme == "tree":
                # Two-hot 4+4: [BRD_OFF:BRD_OFF+4] = super, [BRD_OFF+4:BRD_OFF+8] = fine.
                super_idx, fine_idx = self._board_bucket_tree(
                    my_cards, visible_board)
                state[BRD_OFF + super_idx] = 1.0
                state[BRD_OFF + 4 + fine_idx] = 1.0
            elif self.bucket_scheme == "super":
                # K=4 super-category one-hot at [BRD_OFF:BRD_OFF+4].
                # [BRD_OFF+4:BRD_OFF+8] left as zero — drop-in v3 arch.
                super_idx, _ = self._board_bucket_tree(my_cards, visible_board)
                state[BRD_OFF + super_idx] = 1.0
            elif self.bucket_scheme == "tree42":
                # Two-hot 4+2: super [BRD_OFF:BRD_OFF+4], fine [BRD_OFF+4:BRD_OFF+6].
                # [BRD_OFF+6:BRD_OFF+8] left as zero. fine_2bin compresses the
                # 4-way fine of "tree" scheme into 2 bins (low/high half within super):
                #   HC:  top ≤ T (0) | top ≥ J (1)
                #   Pair: rank ≤ 8 (0) | rank ≥ 9 (1)
                #   MadeNonPaired: TwoPair/Trips (0) | Straight/Flush (1)
                #   Premium: FH/Quads (0) | SF (1)
                super_idx, fine_idx_4 = self._board_bucket_tree(
                    my_cards, visible_board)
                # fine_idx_4 ∈ {0,1,2,3}; collapse to 2-bin by half-split.
                fine_idx_2 = 0 if fine_idx_4 <= 1 else 1
                state[BRD_OFF + super_idx] = 1.0
                state[BRD_OFF + 4 + fine_idx_2] = 1.0
            else:
                # Flat: K_BOARD-way one-hot. Bit-identical with C++.
                brd_bucket = self._board_bucket(my_cards, visible_board)
                state[BRD_OFF + brd_bucket] = 1.0

        # Street one-hot
        street = {0: 0, 3: 1, 4: 2, 5: 3}.get(n_visible, 0)
        state[STREET_OFF + street] = 1.0

        # Betting scalars
        pot       = st["pot"]
        to_call   = st["to_call"]
        my_stack  = st["stacks"][player]
        opp_stack = st["stacks"][1 - player]
        state[BET_OFF + 0] = pot / self.norm
        state[BET_OFF + 1] = to_call / self.norm
        state[BET_OFF + 2] = my_stack / self.starting_stack
        state[BET_OFF + 3] = opp_stack / self.starting_stack

        # Action history
        # 'f' encodes as 0.0 to match C++ FOLD_OR_CHECK slot
        actions = history[3:]
        for i, a in enumerate(actions[-self.HISTORY_LEN:]):
            if isinstance(a, str):
                enc = self.ACTION_ENC.get(a, 0.0)
                if a == 'f':
                    enc = 0.0
                state[HIST_OFF + i] = enc
            else:
                state[HIST_OFF + i] = min(a / self.starting_stack, 1.0)

        # Continuous hand features
        state[CONT_OFF + 0] = equity
        state[CONT_OFF + 1] = board_str
        pot_plus_call = pot + to_call
        state[CONT_OFF + 2] = to_call / pot_plus_call if pot_plus_call > 1e-6 else 0.0
        eff_stack = min(my_stack, opp_stack)
        state[CONT_OFF + 3] = min(eff_stack / pot, 10.0) / 10.0 if pot > 1e-6 else 1.0

        # Quantise continuous dims to stable grouping grid.
        # NOTE: numpy.round uses BANKER'S rounding (0.5 → 0, 1.5 → 2) whereas
        # C++ std::round uses round-half-away-from-zero. On 0.05 grid, raw
        # values commonly fall exactly on N + 0.5 multiples → the two diverge.
        # floor(x/q + 0.5) is unambiguous on both sides (round-half-up); C++
        # uses identical floor((x/q) + 0.5) for parity.
        # Quantise the betting scalars, action history, and continuous tail —
        # everything from BET_OFF to the end.
        cont = slice(BET_OFF, self.state_size())
        state[cont] = np.floor(
            state[cont] / self.FEATURE_QUANT + 0.5
        ) * self.FEATURE_QUANT
        return state.astype(np.float32)