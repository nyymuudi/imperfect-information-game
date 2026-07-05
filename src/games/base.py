"""
Abstract base class for extensive-form imperfect information games.

Any game implementing this interface can be solved by any solver in
the solvers/ module. The interface is intentionally minimal — a game
must only define how to generate information sets, enumerate legal
actions, and evaluate terminal payoffs.

This design follows the standard extensive-form game formalism:
    G = (N, A, H, Z, τ, ρ, σ, u)
where:
    N = set of players
    A = action set
    H = non-terminal histories
    Z = terminal histories
    τ = player function (whose turn)
    ρ = chance probabilities
    σ = information partition
    u = utility function
"""

from abc import ABC, abstractmethod


# Hashable, immutable game state identifier
InfoSetKey = str
Action = str
History = tuple


class ExtensiveFormGame(ABC):
    """
    Abstract extensive-form game with imperfect information.
    
    Subclasses must implement all abstract methods. The solver
    interacts with the game exclusively through this interface,
    ensuring complete domain-agnosticism.
    """

    @abstractmethod
    def num_players(self) -> int:
        """Return the number of players."""
        ...

    @abstractmethod
    def initial_histories(self) -> list[tuple[History, float]]:
        """
        Return all possible initial game states after chance events,
        each paired with its probability.
        
        For Kuhn poker: all possible card dealings.
        For games without initial chance: return [( (), 1.0 )].
        """
        ...

    @abstractmethod
    def is_terminal(self, history: History) -> bool:
        """Return True if history is a terminal (leaf) node."""
        ...

    @abstractmethod
    def terminal_payoffs(self, history: History) -> tuple[float, ...]:
        """
        Return payoff tuple for a terminal history.
        Raises ValueError if history is not terminal.
        """
        ...

    @abstractmethod
    def current_player(self, history: History) -> int:
        """Return the index of the player to act at this history node."""
        ...

    @abstractmethod
    def info_set_key(self, history: History, player: int) -> InfoSetKey:
        """
        Return the information set key for the given player at this history.
        
        Two histories map to the same key iff they are indistinguishable
        to the player — i.e., they belong to the same information set.
        """
        ...

    @abstractmethod
    def legal_actions(self, history: History) -> list[Action]:
        """Return list of legal actions at this history node."""
        ...

    @abstractmethod
    def apply_action(self, history: History, action: Action) -> History:
        """Return new history after applying action."""
        ...